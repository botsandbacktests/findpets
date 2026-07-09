"""Tiny email sender using Gmail SMTP (standard library only).

Set two environment variables (same Gmail App Password you already use for
your trading bot is fine):

    GMAIL_USER          e.g. botsandbacktests@gmail.com
    GMAIL_APP_PASSWORD  the 16-char Google App Password (no spaces)

Optional:
    MAIL_FROM_NAME      display name on the From line (default "FindMyPet")

If the vars aren't set, send_email() returns False and logs a warning instead
of crashing — so the app still runs (useful in local/dev).
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
MAIL_FROM_NAME = os.environ.get("MAIL_FROM_NAME", "FindMyPet")


def mail_configured() -> bool:
    return bool(GMAIL_USER and GMAIL_APP_PASSWORD)


def send_email(to_addr: str, subject: str, text_body: str, html_body: str | None = None) -> bool:
    """Send one email via Gmail SMTP. Returns True on success, False otherwise."""
    print(f"[mailer] send_email() called → to={to_addr!r} "
          f"user_set={bool(GMAIL_USER)} pass_set={bool(GMAIL_APP_PASSWORD)} "
          f"pass_len={len(GMAIL_APP_PASSWORD)}", flush=True)
    if not mail_configured():
        print("[mailer] GMAIL_USER / GMAIL_APP_PASSWORD not set — skipping send.", flush=True)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{MAIL_FROM_NAME} <{GMAIL_USER}>"
    msg["To"] = to_addr
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    try:
        ctx = ssl.create_default_context()
        # Gmail: SSL on 465 (simplest), or STARTTLS on 587.
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"[mailer] SENT OK → {to_addr}", flush=True)
        return True
    except Exception as e:
        print(f"[mailer] send FAILED: {type(e).__name__}: {e}", flush=True)
        return False
