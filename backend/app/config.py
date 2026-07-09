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
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Notify owner when a sighting scores at/above this (0-1) unless they set their own.
DEFAULT_ALERT_THRESHOLD = 0.60
# Max distance (km) considered when matching, unless overridden per-request.
DEFAULT_RADIUS_KM = 10.0

# --- Payments / contact unlock ---
# Your Square hosted payment link (public — safe to keep here).
SQUARE_PAYMENT_LINK = os.environ.get(
    "SQUARE_PAYMENT_LINK", "https://square.link/u/xzvRGMYR"
)
UNLOCK_PRICE_USD = 9.99
UNLOCK_DAYS = 30  # a paid unlock grants contact access for this many days (per pet)

# Secret used to sign login/session tokens. Override in production via env var.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-change-me")

# Public URL of the front-end (used to build password-reset links in emails).
# e.g. https://tech956.com/findpets  — no trailing slash.
SITE_URL = os.environ.get("SITE_URL", "https://tech956.com/findpets").rstrip("/")
