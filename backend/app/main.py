"""FindMyPet API — FastAPI backend with accounts, privacy-gated matches, and paid unlocks."""
from __future__ import annotations

import uuid
import datetime as dt
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .config import (UPLOAD_DIR, DEFAULT_ALERT_THRESHOLD, SQUARE_PAYMENT_LINK, SITE_URL,
                     UNLOCK_PRICE_USD, UNLOCK_DAYS)
from .storage import save_photo, photo_ref_to_url
from .db import SessionLocal, User, Pet, Sighting, ContactUnlock, MatchAlert, init_db
from .embedder import get_embedder
from .matching import find_matches
from .auth import (hash_password, verify_password, make_token, read_token,
                   make_reset_token, read_reset_token)
from .mailer import send_email, mail_configured
from .alerts import send_owner_alert, send_finder_alert

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

app = FastAPI(title="FindMyPet API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    init_db()
    get_embedder()


app.mount("/photos", StaticFiles(directory=str(UPLOAD_DIR)), name="photos")


@app.get("/")
def home():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "FindMyPet API running. Web UI not found at web/index.html."}


# ---------------------------------------------------------------- db + auth deps
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """Resolve the logged-in user from a 'Bearer <token>' header, or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    uid = read_token(authorization[7:])
    if uid is None:
        return None
    return db.get(User, uid)


def require_user(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(401, "Login required")
    return user


# ---------------------------------------------------------------- helpers
def _save_upload(file: UploadFile) -> tuple[str, bytes]:
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    ext = Path(file.filename or "").suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    # Returns a Cloudinary URL in production, or a bare filename locally.
    ref = save_photo(raw, ext)
    return ref, raw


def _save_optional(file: "UploadFile | None") -> str:
    """Save a second/optional photo if provided; return its ref or ''.

    Some clients send an empty file part when no 2nd photo is chosen, so we
    guard on both None and empty bytes.
    """
    if file is None or not getattr(file, "filename", ""):
        return ""
    try:
        ref, _ = _save_upload(file)
        return ref
    except HTTPException:
        return ""  # empty/invalid 2nd photo shouldn't fail the whole request


def _has_active_unlock(db: Session, user_id: int, pet_id: int) -> ContactUnlock | None:
    rows = db.query(ContactUnlock).filter(
        ContactUnlock.user_id == user_id, ContactUnlock.pet_id == pet_id
    ).all()
    for r in rows:
        if r.is_active():
            return r
    return None


def _pet_match_view(pet: Pet, distance_km: float, score_pct: float) -> dict:
    """
    PRIVACY-GATED view returned in match results BEFORE payment.
    Deliberately contains NO free text (no name, breed text, description) and
    NO contact info — only the photo, similarity score, distance, and coarse
    non-identifying attributes. This prevents anyone from smuggling contact
    details into a text field to bypass the paywall.
    """
    return {
        "id": pet.id,
        "species": pet.species,          # coarse category only
        "size": pet.size,                # coarse category only
        "photo_url": photo_ref_to_url(pet.photo_path),
        "distance_km": round(distance_km, 2),
        "score_pct": round(score_pct, 1),
        "last_seen_at": pet.last_seen_at.isoformat(),
    }


def _pet_owner_view(pet: Pet) -> dict:
    """Full view for the pet's own owner (their own data)."""
    return {
        "id": pet.id, "name": pet.name, "species": pet.species, "breed": pet.breed,
        "color": pet.color, "size": pet.size, "description": pet.description,
        "status": pet.status, "photo_url": photo_ref_to_url(pet.photo_path),
        "last_seen_at": pet.last_seen_at.isoformat(),
    }


# ---------------------------------------------------------------- auth routes
@app.post("/api/auth/signup")
def signup(email: str = Form(...), password: str = Form(...),
           display_name: str = Form(""), db: Session = Depends(get_db)):
    email = email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "An account with that email already exists.")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    u = User(email=email, display_name=display_name, password_hash=hash_password(password))
    db.add(u); db.commit(); db.refresh(u)
    return {"token": make_token(u.id), "email": u.email, "display_name": u.display_name}


@app.post("/api/auth/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    email = email.strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(password, u.password_hash):
        raise HTTPException(401, "Invalid email or password.")
    return {"token": make_token(u.id), "email": u.email, "display_name": u.display_name}


@app.get("/api/auth/me")
def me(user: User = Depends(require_user)):
    return {"id": user.id, "email": user.email, "display_name": user.display_name}


@app.post("/api/auth/forgot")
def forgot_password(email: str = Form(...), db: Session = Depends(get_db)):
    """Email a password-reset link if the account exists.

    Always returns the same success message whether or not the email exists,
    so attackers can't use this to discover which emails are registered.
    """
    email = email.strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if u:
        token = make_reset_token(u.id, u.password_hash, minutes=30)
        link = f"{SITE_URL}/reset.html?token={token}"
        text = (
            f"Hi,\n\nWe got a request to reset your FindMyPet password.\n\n"
            f"Click this link to set a new password (valid for 30 minutes):\n{link}\n\n"
            f"If you didn't request this, you can ignore this email — your "
            f"password won't change.\n\n— FindMyPet"
        )
        html = (
            f"<p>Hi,</p><p>We got a request to reset your <b>FindMyPet</b> password.</p>"
            f"<p><a href=\"{link}\">Click here to set a new password</a> "
            f"(valid for 30 minutes).</p>"
            f"<p>If you didn't request this, you can ignore this email — your "
            f"password won't change.</p>"
            f"<p style=\"color:#6b7280;font-size:13px\">Tip: add us to your contacts so future "
            f"FindMyPet alerts don't land in spam.</p><p>— FindMyPet</p>"
        )
        send_email(u.email, "Reset your FindMyPet password", text, html)
    return {"ok": True,
            "message": "If an account exists for that email, a reset link has been sent."}


@app.post("/api/auth/reset")
def reset_password(token: str = Form(...), new_password: str = Form(...),
                   db: Session = Depends(get_db)):
    """Set a new password using a valid reset token from the emailed link."""
    if len(new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    # We need the user's CURRENT hash to validate the token, but the token only
    # carries the uid inside its signed body. Decode uid loosely first.
    try:
        import base64, json
        body = token.split(".")[0]
        pad = "=" * (-len(body) % 4)
        uid = int(json.loads(base64.urlsafe_b64decode(body + pad))["uid"])
    except Exception:
        raise HTTPException(400, "This reset link is invalid or has expired.")
    u = db.get(User, uid)
    if not u or read_reset_token(token, u.password_hash) != u.id:
        raise HTTPException(400, "This reset link is invalid or has expired.")
    u.password_hash = hash_password(new_password)
    db.commit()
    # Log them straight in after resetting.
    return {"ok": True, "token": make_token(u.id), "email": u.email,
            "display_name": u.display_name}


# ---------------------------------------------------------------- pets / sightings
@app.get("/api/health")
def health():
    emb = get_embedder()
    from .storage import storage_backend
    from .config import DATABASE_URL
    db_kind = "postgres" if DATABASE_URL.startswith("postgresql") else "sqlite"
    return {"status": "ok", "embedder": emb.name, "dim": emb.dim,
            "database": db_kind, "photo_storage": storage_backend(),
            "unlock_price_usd": UNLOCK_PRICE_USD, "unlock_days": UNLOCK_DAYS}


@app.post("/api/pets")
def report_lost_pet(
    name: str = Form(...), species: str = Form(...), breed: str = Form(""),
    color: str = Form(""), size: str = Form(""), description: str = Form(""),
    last_seen_lat: float = Form(...), last_seen_lng: float = Form(...),
    alert_radius_km: float = Form(16.0), alert_threshold: float = Form(DEFAULT_ALERT_THRESHOLD),
    contact_name: str = Form(""), contact_email: str = Form(""), contact_phone: str = Form(""),
    photo: UploadFile = File(...),
    photo2: UploadFile | None = File(None),
    user: User = Depends(require_user), db: Session = Depends(get_db),
):
    fname, raw = _save_upload(photo)
    fname2 = _save_optional(photo2)
    emb = get_embedder()
    vec = emb.embed(raw)
    pet = Pet(
        owner_id=user.id, name=name, species=species, breed=breed, color=color,
        size=size, description=description, status="lost",
        last_seen_lat=last_seen_lat, last_seen_lng=last_seen_lng,
        last_seen_at=dt.datetime.utcnow(),
        alert_radius_km=alert_radius_km, alert_threshold=alert_threshold,
        contact_name=contact_name or user.display_name,
        contact_email=contact_email or user.email, contact_phone=contact_phone,
        photo_path=fname, photo_path2=fname2,
        embedding=vec.astype(np.float32).tobytes(), embed_model=emb.name,
    )
    db.add(pet); db.commit(); db.refresh(pet)
    return _pet_owner_view(pet)


@app.get("/api/pets/mine")
def my_pets(user: User = Depends(require_user), db: Session = Depends(get_db)):
    pets = db.query(Pet).filter(Pet.owner_id == user.id).order_by(Pet.created_at.desc()).all()
    return [_pet_owner_view(p) for p in pets]


@app.post("/api/pets/{pet_id}/found")
def mark_found(pet_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(404, "Pet not found")
    if pet.owner_id != user.id:
        raise HTTPException(403, "You can only update your own pet.")
    pet.status = "found"; db.commit()
    return {"id": pet.id, "status": pet.status}


def _dispatch_match_alerts(
    db: Session, matches: list[dict], sighting_id: int | None,
    finder_email: str, sighting_photo_urls: list[str] | None = None,
) -> None:
    """Best-effort: email owners (and the finder) about qualifying matches.

    - Emails a pet's owner when the match score >= that pet's alert_threshold.
    - Emails the finder once if they provided a contact_email.
    - Skips any (pet, sighting) pair already alerted (de-dup via MatchAlert).
    - Never raises: a mail problem must not break the match response.
    """
    finder_email = (finder_email or "").strip().lower()
    finder_notified = False
    try:
        for m in matches:
            pet: Pet = m["pet"]
            sim = m["similarity"]              # 0-1 cosine similarity
            score_pct = m["score_pct"]
            distance_km = m["distance_km"]

            threshold = pet.alert_threshold or DEFAULT_ALERT_THRESHOLD
            if sim < threshold:
                continue  # not a strong enough match to bother the owner

            # De-dup: have we already alerted for this pet + sighting?
            already = None
            if sighting_id is not None:
                already = db.query(MatchAlert).filter(
                    MatchAlert.pet_id == pet.id,
                    MatchAlert.sighting_id == sighting_id,
                ).first()
            if already:
                continue

            owner_email = (pet.contact_email or "").strip()
            if not owner_email and pet.owner_id:
                owner = db.get(User, pet.owner_id)
                owner_email = owner.email if owner else ""

            owner_ok = send_owner_alert(
                owner_email, pet.name, score_pct, distance_km, SITE_URL,
                sighting_photo_urls=sighting_photo_urls,
            ) if owner_email else False

            finder_ok = False
            if finder_email and not finder_notified:
                finder_ok = send_finder_alert(finder_email, score_pct, SITE_URL)
                finder_notified = finder_ok or finder_notified

            db.add(MatchAlert(
                pet_id=pet.id, sighting_id=sighting_id,
                owner_emailed=1 if owner_ok else 0,
                finder_emailed=1 if finder_ok else 0,
                score_pct=score_pct,
            ))
        db.commit()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[alerts] dispatch failed (non-fatal): {type(e).__name__}: {e}",
              flush=True)
        db.rollback()


@app.post("/api/sightings/match")
def report_sighting_and_match(
    lat: float = Form(...), lng: float = Form(...),
    search_radius_km: float = Form(16.0), species: str = Form(""),
    contact_name: str = Form(""), contact_email: str = Form(""), contact_phone: str = Form(""),
    save: bool = Form(True), photo: UploadFile = File(...),
    photo2: UploadFile | None = File(None),
    user: User | None = Depends(current_user), db: Session = Depends(get_db),
):
    """Upload a spotted animal, embed it, return PRIVACY-GATED ranked matches."""
    fname, raw = _save_upload(photo)
    fname2 = _save_optional(photo2)
    emb = get_embedder()
    vec = emb.embed(raw)
    matches = find_matches(db, query_vec=vec, query_model=emb.name,
                           lat=lat, lng=lng, radius_km=search_radius_km,
                           species=species or None, top_k=10)
    sighting_id = None
    if save:
        s = Sighting(
            reporter_id=user.id if user else None,
            lat=lat, lng=lng, search_radius_km=search_radius_km,
            status="matched" if matches else "open",
            contact_name=contact_name, contact_email=contact_email, contact_phone=contact_phone,
            photo_path=fname, photo_path2=fname2,
            embedding=vec.astype(np.float32).tobytes(), embed_model=emb.name,
        )
        db.add(s); db.commit()
        sighting_id = s.id

    # Notify owners (and the finder) about qualifying matches. Best-effort:
    # this must never break the response, so failures are swallowed inside.
    if matches:
        sighting_photo_urls = [
            photo_ref_to_url(ref) for ref in (fname, fname2) if ref
        ]
        _dispatch_match_alerts(
            db, matches, sighting_id, contact_email,
            sighting_photo_urls=sighting_photo_urls,
        )

    out = []
    for m in matches:
        pet = m["pet"]
        view = _pet_match_view(pet, m["distance_km"], m["score_pct"])
        # Tell a logged-in user whether they've already unlocked this pet.
        view["contact_unlocked"] = bool(user and _has_active_unlock(db, user.id, pet.id))
        out.append(view)

    return {"embedder": emb.name, "match_count": len(out),
            "unlock_price_usd": UNLOCK_PRICE_USD, "matches": out}


# ---------------------------------------------------------------- unlock flow
@app.post("/api/unlock/start")
def unlock_start(pet_id: int = Form(...),
                 user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Begin an unlock. Returns the Square payment link to send the user to."""
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(404, "Pet not found")
    existing = _has_active_unlock(db, user.id, pet_id)
    if existing:
        return {"already_active": True, "expires_at": existing.expires_at.isoformat()}
    u = ContactUnlock(user_id=user.id, pet_id=pet_id, status="pending",
                      amount_usd=UNLOCK_PRICE_USD)
    db.add(u); db.commit(); db.refresh(u)
    return {
        "unlock_id": u.id,
        "payment_link": SQUARE_PAYMENT_LINK,
        "price_usd": UNLOCK_PRICE_USD,
        "instructions": ("Complete payment on the Square page, then return here and "
                         "confirm using the email or receipt number from your Square receipt."),
    }


@app.post("/api/unlock/confirm")
def unlock_confirm(unlock_id: int = Form(...), payment_ref: str = Form(...),
                   tip_usd: float = Form(0.0),
                   user: User = Depends(require_user), db: Session = Depends(get_db)):
    """
    Confirm a payment and activate the 30-day pass.

    NOTE: With a basic Square payment link we cannot automatically verify the
    payment server-side — this records the user's payment reference and activates
    the pass. When deployed, this is replaced by a Square webhook that verifies
    the payment automatically before activation. (See README.)
    """
    u = db.get(ContactUnlock, unlock_id)
    if not u or u.user_id != user.id:
        raise HTTPException(404, "Unlock not found")
    if u.is_active():
        return {"status": "active", "expires_at": u.expires_at.isoformat()}
    now = dt.datetime.utcnow()
    u.status = "active"
    u.payment_ref = payment_ref
    u.tip_usd = max(0.0, tip_usd)
    u.activated_at = now
    u.expires_at = now + dt.timedelta(days=UNLOCK_DAYS)
    db.commit()
    return {"status": "active", "expires_at": u.expires_at.isoformat()}


@app.get("/api/pets/{pet_id}/contact")
def get_contact(pet_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Return contact info ONLY if the user holds an active unlock for this pet."""
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(404, "Pet not found")
    unlock = _has_active_unlock(db, user.id, pet_id)
    if not unlock:
        raise HTTPException(402, "Payment required to view contact info.")
    return {
        "pet_id": pet.id,
        "name": pet.name,
        "breed": pet.breed,
        "color": pet.color,
        "description": pet.description,
        "contact_name": pet.contact_name,
        "contact_email": pet.contact_email,
        "contact_phone": pet.contact_phone,
        "pass_expires_at": unlock.expires_at.isoformat(),
    }
