"""FindMyPet API — FastAPI backend with accounts, privacy-gated matches, and paid unlocks."""
from __future__ import annotations

import uuid
import datetime as dt
from pathlib import Path

import numpy as np
from fastapi import (FastAPI, UploadFile, File, Form, HTTPException, Depends,
                     Header, Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .config import (UPLOAD_DIR, DEFAULT_ALERT_THRESHOLD, SQUARE_PAYMENT_LINK, SITE_URL,
                     UNLOCK_PRICE_USD, UNLOCK_DAYS, FINDER_UNLOCK_PRICE_USD,
                     BUNDLE_MATCH_MIN_PCT, square_api_configured)
from .storage import save_photo, photo_ref_to_url
from .db import (SessionLocal, User, Pet, Sighting, ContactUnlock, FinderUnlock,
                 MatchAlert, init_db)
from .embedder import get_embedder, embed_with_fallback
from .matching import find_matches
from .auth import (hash_password, verify_password, make_token, read_token,
                   make_reset_token, read_reset_token)
from .mailer import send_email, mail_configured
from .alerts import send_owner_alert, send_finder_alert
from . import square_client as sq

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


def _has_active_finder_unlock(db: Session, user_id: int, pet_id: int) -> FinderUnlock | None:
    """Active PER-PET bundle pass for this owner+pet, or None."""
    rows = db.query(FinderUnlock).filter(
        FinderUnlock.user_id == user_id, FinderUnlock.pet_id == pet_id
    ).all()
    for r in rows:
        if r.is_active():
            return r
    return None


def _score_sighting_against_pet(pet: Pet, s: Sighting) -> tuple[float, float] | None:
    """Return (similarity_pct, distance_km) for a sighting vs a pet, or None if
    they can't be compared (different embed model) or the sighting is out of its
    own search radius. Used by both the owner's sighting list and the contact gate
    so the qualifying rule is identical in both places.
    """
    from .embedder import cosine_similarity
    from .geo import haversine_km
    if not s.embedding or s.embed_model != pet.embed_model:
        return None
    dist = haversine_km(pet.last_seen_lat, pet.last_seen_lng, s.lat, s.lng)
    if dist > (s.search_radius_km or 16.0):
        return None
    pet_vec = np.frombuffer(pet.embedding, dtype=np.float32)
    sim = cosine_similarity(pet_vec, np.frombuffer(s.embedding, dtype=np.float32))
    return round(sim * 100, 1), round(dist, 2)


def _sighting_gated_view(s: Sighting, pet_id: int, score_pct: float,
                         distance_km: float, unlocked: bool) -> dict:
    """PRIVACY-GATED view of a sighting shown to a pet's owner BEFORE payment.

    Shows the spotted-animal photos (proof it might be their pet) + score +
    distance, but NO finder contact info. Contact is revealed only via
    /api/sightings/{id}/contact once the owner holds an active FinderUnlock
    for the pet AND the sighting scores >= BUNDLE_MATCH_MIN_PCT.
    """
    return {
        "sighting_id": s.id,
        "matched_pet_id": pet_id,
        "photo_url": photo_ref_to_url(s.photo_path),
        "photo_url2": photo_ref_to_url(s.photo_path2) if s.photo_path2 else "",
        "score_pct": round(score_pct, 1),
        "distance_km": round(distance_km, 2),
        "seen_at": s.created_at.isoformat(),
        "has_finder_contact": bool(s.contact_email or s.contact_phone or s.contact_name),
        "finder_unlocked": unlocked,
    }


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
            "unlock_price_usd": UNLOCK_PRICE_USD, "unlock_days": UNLOCK_DAYS,
            "finder_unlock_price_usd": FINDER_UNLOCK_PRICE_USD,
            # "auto" = unique links + webhook verification; "manual" = static link
            "payment_verify": "auto" if square_api_configured() else "manual"}


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
    vec, model_name = embed_with_fallback(raw)
    pet = Pet(
        owner_id=user.id, name=name, species=species, breed=breed, color=color,
        size=size, description=description, status="lost",
        last_seen_lat=last_seen_lat, last_seen_lng=last_seen_lng,
        last_seen_at=dt.datetime.utcnow(),
        alert_radius_km=alert_radius_km, alert_threshold=alert_threshold,
        contact_name=contact_name or user.display_name,
        contact_email=contact_email or user.email, contact_phone=contact_phone,
        photo_path=fname, photo_path2=fname2,
        embedding=vec.astype(np.float32).tobytes(), embed_model=model_name,
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
    vec, model_name = embed_with_fallback(raw)
    matches = find_matches(db, query_vec=vec, query_model=model_name,
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
            embedding=vec.astype(np.float32).tobytes(), embed_model=model_name,
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

    return {"embedder": model_name, "match_count": len(out),
            "unlock_price_usd": UNLOCK_PRICE_USD, "matches": out}


# ---------------------------------------------------------------- unlock flow
def _checkout_for(kind: str, unlock_id: int, amount_usd: float, name: str) -> dict:
    """Return the payment link + how it was verified.

    When the Square API is configured, mint a UNIQUE hosted checkout carrying our
    unlock ref so the webhook can auto-verify. Otherwise fall back to the static
    link + manual receipt entry. Never raises — a Square hiccup just degrades to
    the static link so the user can still pay.
    """
    if square_api_configured():
        try:
            link = sq.create_payment_link(kind, unlock_id, amount_usd, name)
            if link.get("url"):
                return {"payment_link": link["url"], "verify": "auto",
                        "payment_link_id": link.get("payment_link_id", "")}
        except sq.SquareError as e:
            print(f"[square] create link failed, using static link: {e}", flush=True)
    return {"payment_link": SQUARE_PAYMENT_LINK, "verify": "manual"}


@app.post("/api/unlock/start")
def unlock_start(pet_id: int = Form(...),
                 user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Begin a finder->owner unlock. Returns a (unique when possible) Square link."""
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(404, "Pet not found")
    existing = _has_active_unlock(db, user.id, pet_id)
    if existing:
        return {"already_active": True, "expires_at": existing.expires_at.isoformat()}
    u = ContactUnlock(user_id=user.id, pet_id=pet_id, status="pending",
                      amount_usd=UNLOCK_PRICE_USD)
    db.add(u); db.commit(); db.refresh(u)
    checkout = _checkout_for("pet", u.id, UNLOCK_PRICE_USD,
                             f"FindMyPet — unlock owner contact")
    return {
        "unlock_id": u.id,
        "kind": "pet",
        "payment_link": checkout["payment_link"],
        "verify": checkout["verify"],   # "auto" = webhook; "manual" = type receipt
        "price_usd": UNLOCK_PRICE_USD,
        "instructions": ("Complete payment on the Square page. We'll confirm it "
                         "automatically and unlock the contact."),
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


# --------------------------------------------- owner-pays-to-see-finder unlock flow
# This is a PER-PET BUNDLE: the owner pays FINDER_UNLOCK_PRICE_USD ($24.99) once
# for a pet and that unlocks the finder contact on EVERY sighting that scores
# >= BUNDLE_MATCH_MIN_PCT (65% = green + orange) against that pet, for 30 days,
# including new qualifying sightings that arrive during the window.
@app.get("/api/pets/{pet_id}/sightings")
def sightings_for_my_pet(pet_id: int, user: User = Depends(require_user),
                         db: Session = Depends(get_db)):
    """List sightings that MATCH one of my lost pets, so I (the owner) can see who
    spotted my pet and pay once to reveal ALL their finder contacts.

    Only the pet's own owner may call this. Returns privacy-gated views (photos +
    score + distance, no contact). If I hold an active per-pet pass, every
    qualifying (>=65%) sighting is marked unlocked.
    """
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(404, "Pet not found")
    if pet.owner_id != user.id:
        raise HTTPException(403, "You can only view sightings for your own pet.")

    has_pass = bool(_has_active_finder_unlock(db, user.id, pet.id))
    sightings = db.query(Sighting).order_by(Sighting.created_at.desc()).all()
    out = []
    for s in sightings:
        scored = _score_sighting_against_pet(pet, s)
        if scored is None:
            continue
        score_pct, dist = scored
        if score_pct < 50:
            continue  # hide clearly-irrelevant matches from the list entirely
        qualifies = score_pct >= BUNDLE_MATCH_MIN_PCT  # covered by the bundle?
        unlocked = has_pass and qualifies
        view = _sighting_gated_view(s, pet.id, score_pct, dist, unlocked)
        view["qualifies_for_bundle"] = qualifies
        out.append(view)

    out.sort(key=lambda r: r["score_pct"], reverse=True)
    return {
        "pet_id": pet.id,
        "match_count": len(out),
        "qualifying_count": sum(1 for v in out if v["qualifies_for_bundle"]),
        "finder_unlock_price_usd": FINDER_UNLOCK_PRICE_USD,
        "bundle_min_pct": BUNDLE_MATCH_MIN_PCT,
        "has_pass": has_pass,
        "sightings": out,
    }


@app.post("/api/finder-unlock/start")
def finder_unlock_start(pet_id: int = Form(...),
                        user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Begin a per-pet owner->finder bundle unlock. Returns the Square link."""
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(404, "Pet not found")
    if pet.owner_id != user.id:
        raise HTTPException(403, "You can only unlock finders for your own pet.")
    existing = _has_active_finder_unlock(db, user.id, pet_id)
    if existing:
        return {"already_active": True, "expires_at": existing.expires_at.isoformat()}
    u = FinderUnlock(user_id=user.id, pet_id=pet_id, status="pending",
                     amount_usd=FINDER_UNLOCK_PRICE_USD)
    db.add(u); db.commit(); db.refresh(u)
    checkout = _checkout_for("finder", u.id, FINDER_UNLOCK_PRICE_USD,
                             f"FindMyPet — unlock finder contacts")
    return {
        "unlock_id": u.id,
        "kind": "finder",
        "payment_link": checkout["payment_link"],
        "verify": checkout["verify"],
        "price_usd": FINDER_UNLOCK_PRICE_USD,
        "instructions": ("Complete payment on the Square page. We'll confirm it "
                         "automatically and unlock the finder contacts."),
    }


@app.post("/api/finder-unlock/confirm")
def finder_unlock_confirm(unlock_id: int = Form(...), payment_ref: str = Form(...),
                          tip_usd: float = Form(0.0),
                          user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Confirm payment and activate the 30-day per-pet finder-contact pass.

    Same manual-confirm caveat as /api/unlock/confirm: a basic Square link can't
    be verified server-side, so this records the reference and activates the pass.
    Replaced by a Square webhook when deployed. (See README.)
    """
    u = db.get(FinderUnlock, unlock_id)
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


@app.get("/api/sightings/{sighting_id}/contact")
def get_finder_contact(sighting_id: int, pet_id: int,
                       user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Return a finder's contact info ONLY if the user holds an active per-pet
    bundle pass for `pet_id` AND this sighting scores >= BUNDLE_MATCH_MIN_PCT (65%)
    against that pet. The pet must be the caller's own pet.
    """
    s = db.get(Sighting, sighting_id)
    if not s:
        raise HTTPException(404, "Sighting not found")
    pet = db.get(Pet, pet_id)
    if not pet or pet.owner_id != user.id:
        raise HTTPException(403, "You can only view finders for your own pet.")

    unlock = _has_active_finder_unlock(db, user.id, pet_id)
    if not unlock:
        raise HTTPException(402, "Payment required to view finder contact info.")

    scored = _score_sighting_against_pet(pet, s)
    if scored is None or scored[0] < BUNDLE_MATCH_MIN_PCT:
        # Pass is valid but this particular sighting isn't a strong enough match
        # to be part of the bundle.
        raise HTTPException(403, "This sighting isn't a strong enough match to unlock.")

    return {
        "sighting_id": s.id,
        "contact_name": s.contact_name,
        "contact_email": s.contact_email,
        "contact_phone": s.contact_phone,
        "note": s.note,
        "pass_expires_at": unlock.expires_at.isoformat(),
    }


# ------------------------------------------------ Square webhook + payment status
def _activate_unlock(db: Session, kind: str, unlock_id: int, payment_ref: str,
                     tip_usd: float = 0.0) -> bool:
    """Activate a ContactUnlock ('pet') or FinderUnlock ('finder') by id.

    Idempotent: re-running on an already-active pass is a no-op that returns True,
    so a re-delivered webhook can't double-charge time or error out. Returns True
    if the unlock exists and is now active.
    """
    Model = ContactUnlock if kind == "pet" else FinderUnlock
    u = db.get(Model, unlock_id)
    if not u:
        return False
    if u.is_active():
        return True
    now = dt.datetime.utcnow()
    u.status = "active"
    u.payment_ref = payment_ref or u.payment_ref
    u.tip_usd = max(0.0, float(tip_usd or 0.0))
    u.activated_at = now
    u.expires_at = now + dt.timedelta(days=UNLOCK_DAYS)
    db.commit()
    return True


@app.post("/api/square/webhook")
async def square_webhook(request: Request, db: Session = Depends(get_db)):
    """Receive Square payment notifications and auto-activate the matching unlock.

    Security: verifies the x-square-hmacsha256-signature over (notification_url +
    raw body) using our webhook signature key. Unverified calls are rejected.

    Matching: we stamped 'pet:<id>' or 'finder:<id>' into the payment_note when
    creating the checkout, so the payment object in the webhook carries it back.
    Only COMPLETED payments activate an unlock.
    """
    raw = await request.body()
    sig = request.headers.get("x-square-hmacsha256-signature", "")
    if not sq.verify_webhook_signature(raw, sig):
        raise HTTPException(401, "Invalid webhook signature")

    try:
        event = __import__("json").loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "Bad JSON")

    # Dig out the payment object (shape: data.object.payment).
    payment = (((event.get("data") or {}).get("object") or {}).get("payment")) or {}
    status = (payment.get("status") or "").upper()
    note = payment.get("note") or ""
    order_id = payment.get("order_id") or ""
    payment_id = payment.get("id") or ""

    # Recover our unlock ref: prefer the payment note; fall back to the order's
    # reference_id if a note wasn't carried through.
    ref = sq.parse_ref(note)
    if ref is None and order_id:
        ref = sq.parse_ref(sq.get_order_reference(order_id))

    # Acknowledge (200) even when we can't act, so Square stops retrying a payment
    # that isn't ours or isn't finished. We just don't activate anything.
    if status not in ("COMPLETED", "APPROVED", "CAPTURED"):
        return {"ok": True, "ignored": f"status={status or 'unknown'}"}
    if ref is None:
        return {"ok": True, "ignored": "no matching unlock ref"}

    kind, unlock_id = ref
    ok = _activate_unlock(db, kind, unlock_id, payment_ref=payment_id)
    return {"ok": True, "activated": ok, "kind": kind, "unlock_id": unlock_id}


@app.get("/api/unlock/status")
def unlock_status(kind: str, unlock_id: int,
                  user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Let the frontend poll whether a pending unlock has been activated yet.

    Used after the buyer returns from Square: the page polls this until the
    webhook flips the unlock to active, then reveals the contact. Only the owner
    of the unlock can read its status.
    """
    if kind not in ("pet", "finder"):
        raise HTTPException(400, "Bad kind")
    Model = ContactUnlock if kind == "pet" else FinderUnlock
    u = db.get(Model, unlock_id)
    if not u or u.user_id != user.id:
        raise HTTPException(404, "Unlock not found")
    active = u.is_active()
    return {
        "kind": kind,
        "unlock_id": u.id,
        "status": "active" if active else u.status,
        "active": active,
        "expires_at": u.expires_at.isoformat() if u.expires_at else None,
        # echo the target id so the frontend knows what to reveal
        "target_id": getattr(u, "pet_id", None),
    }
