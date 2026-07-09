"""Tiny email sender using the Resend HTTP API (works on Render's free tier).

Render's free plan blocks outbound SMTP (ports 25/465/587), so we send over
HTTPS (port 443) via Resend's REST API instead. Standard library only — no
extra packages.

Set these environment variables in Render:

    RESEND_API_KEY   your Resend API key (starts with "re_")
    MAIL_FROM        the verified sender, e.g. "FindMyPet <noreply@tech956.com>"
                     To test immediately without verifying a domain, use:
                     "FindMyPet <onboarding@resend.dev>"

If RESEND_API_KEY isn't set, send_email() logs a warning and returns False
so the app still runs.
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
# Default sender uses Resend's shared test domain so it works before you verify
# tech956.com. Override MAIL_FROM once your domain is verified in Resend.
MAIL_FROM = os.environ.get("MAIL_FROM", "FindMyPet <onboarding@resend.dev>")

_RESEND_URL = "https://api.resend.com/emails"


def mail_configured() -> bool:
    return bool(RESEND_API_KEY)


def send_email(to_addr: str, subject: str, text_body: str, html_body: str | None = None) -> bool:
    """Send one email via the Resend HTTP API. Returns True on success."""
    print(f"[mailer] send_email() called → to={to_addr!r} "
          f"key_set={bool(RESEND_API_KEY)} from={MAIL_FROM!r}", flush=True)
    if not mail_configured():
        print("[mailer] RESEND_API_KEY not set — skipping send.", flush=True)
        return False

    payload = {
        "from": MAIL_FROM,
        "to": [to_addr],
        "subject": subject,
        "text": text_body,
    }
    if html_body:
        payload["html"] = html_body

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _RESEND_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            # Cloudflare (in front of Resend) blocks the default Python-urllib
            # User-Agent with a 403 / error 1010. Send a normal UA so the
            # request isn't treated as a bot.
            "User-Agent": "FindMyPet/1.0 (+https://tech956.com/findpets)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", "replace")
            print(f"[mailer] SENT OK → {to_addr} (HTTP {resp.status}) {body[:200]}", flush=True)
            return True
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        print(f"[mailer] send FAILED: HTTP {e.code} {detail[:600]}", flush=True)
        return False
    except Exception as e:
        print(f"[mailer] send FAILED: {type(e).__name__}: {e}", flush=True)
        return False
