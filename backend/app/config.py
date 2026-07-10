"""Central configuration for the FindMyPet backend."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_DIR = BASE_DIR.parent                               # FindMyPet/
DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "findmypet.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: SQLite needs a filesystem that supports file locking. Network/synced
# folders (e.g. some cloud-mounted dirs) can raise "disk I/O error". If that
# happens, set DATABASE_URL to a local path, e.g.:
#   export DATABASE_URL="sqlite:////absolute/local/path/findmypet.db"
#
# In PRODUCTION set DATABASE_URL to a Render Postgres URL so data survives
# redeploys. Render hands out URLs starting with "postgres://", but SQLAlchemy
# needs the "postgresql://" scheme — we normalize that below.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Notify owner when a sighting scores at/above this (0-1) unless they set their own.
# Set to 0.75 to match the frontend's "strong match" bar: CLIP scores run warm
# (two dogs alone ~60%), so 0.60 emailed owners about weak cross-breed lookalikes.
# 0.75 only alerts on genuinely likely matches, cutting false-alarm emails.
DEFAULT_ALERT_THRESHOLD = 0.75
# Max distance (km) considered when matching, unless overridden per-request.
DEFAULT_RADIUS_KM = 10.0

# --- Payments / contact unlock ---
# Fallback static Square hosted payment link, used ONLY when the Square API isn't
# configured (below). Once the API creds are set, each unlock gets its OWN unique
# checkout link so the webhook can match the payment back to the exact unlock.
SQUARE_PAYMENT_LINK = os.environ.get(
    "SQUARE_PAYMENT_LINK", "https://square.link/u/xzvRGMYR"
)

# --- Square API (unique checkout links + webhook auto-verification) ---
# All read from Render env vars; blank locally so the app falls back to the
# static link + manual receipt entry. Get these from the Square Developer
# Dashboard (see SETUP-square-payments.md).
SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN", "").strip()
SQUARE_LOCATION_ID = os.environ.get("SQUARE_LOCATION_ID", "").strip()
SQUARE_WEBHOOK_SIGNATURE_KEY = os.environ.get("SQUARE_WEBHOOK_SIGNATURE_KEY", "").strip()
# "sandbox" (test cards, no real money) or "production" (real charges).
SQUARE_ENV = os.environ.get("SQUARE_ENV", "sandbox").strip().lower()
# Public URL Square calls with payment notifications. Must EXACTLY match the URL
# entered in the Square webhook subscription (used in signature verification).
SQUARE_WEBHOOK_URL = os.environ.get(
    "SQUARE_WEBHOOK_URL",
    "https://findmypet-api-9pez.onrender.com/api/square/webhook",
).strip()

def square_api_configured() -> bool:
    """True when we have enough to create checkout links via the Square API."""
    return bool(SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID)

def square_api_base() -> str:
    return ("https://connect.squareupsandbox.com"
            if SQUARE_ENV != "production"
            else "https://connect.squareup.com")

UNLOCK_PRICE_USD = 9.99  # finder pays to see the pet OWNER's contact info
UNLOCK_DAYS = 30  # a paid unlock grants contact access for this many days (per pet)

# Owner pays to see the FINDER's contact info (people who reported sightings of
# their lost pet). This is a PER-PET BUNDLE: one payment unlocks the finder
# contact for EVERY qualifying sighting on that pet for UNLOCK_DAYS (30) days,
# including new qualifying sightings that arrive during the window.
FINDER_UNLOCK_PRICE_USD = float(os.environ.get("FINDER_UNLOCK_PRICE_USD", 24.99))

# A sighting qualifies for the bundle (and is shown to the owner) when it scores
# at/above this % against the pet. 65 = the app's "possible match" (orange) bar,
# so the $24.99 unlocks all green (80%+) AND orange (65-80%) matches. Weaker
# matches (<65%) stay hidden and locked. Keep in sync with app.html scoreClass().
BUNDLE_MATCH_MIN_PCT = 65.0

# Secret used to sign login/session tokens. Override in production via env var.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-change-me")

# Public URL of the front-end (used to build password-reset links in emails).
# e.g. https://tech956.com/findpets  — no trailing slash.
SITE_URL = os.environ.get("SITE_URL", "https://tech956.com/findpets").rstrip("/")

# Where Square sends the buyer AFTER paying — back to the app so it can poll for
# the webhook to confirm. Defaults to the app page with a ?paid=1 flag.
SQUARE_REDIRECT_URL = os.environ.get("SQUARE_REDIRECT_URL", f"{SITE_URL}/app.html?paid=1")

# --- Photo storage (Cloudinary) ---
# When CLOUDINARY_URL is set (in Render env), uploaded pet photos are stored on
# Cloudinary and survive redeploys. If it's blank, we fall back to saving files
# to the local UPLOAD_DIR (fine for local dev; wiped on Render's free disk).
CLOUDINARY_URL = os.environ.get("CLOUDINARY_URL", "").strip()

# --- Real AI matching (hosted CLIP embeddings via Replicate) ---
# When REPLICATE_API_TOKEN is set (in Render env), the app sends each pet photo
# to Replicate's hosted CLIP model and gets back a real 768-dim feature vector —
# actual visual matching, running fine on Render's free 512MB tier.
# If it's blank, embedder.py falls back to local DINOv2 (needs torch) or, last
# resort, the weak color/texture descriptor. Read directly from the environment
# in embedder.py; surfaced here so all config lives in one place.
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "").strip()
