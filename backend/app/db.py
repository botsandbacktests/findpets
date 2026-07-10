"""Database setup and ORM models."""
from __future__ import annotations

import datetime as dt
from sqlalchemy import (create_engine, String, Integer, Float, DateTime,
                        ForeignKey, LargeBinary, Text)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)

from .config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
# pool_pre_ping recycles dead connections — important on Render, where a sleeping
# free instance drops its Postgres connections. Harmless for SQLite.
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    phone: Mapped[str] = mapped_column(String(40), default="")  # optional contact
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Pet(Base):
    __tablename__ = "pets"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    name: Mapped[str] = mapped_column(String(120))
    species: Mapped[str] = mapped_column(String(40))          # dog, cat, ...
    breed: Mapped[str] = mapped_column(String(120), default="")
    color: Mapped[str] = mapped_column(String(80), default="")
    size: Mapped[str] = mapped_column(String(40), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="lost")  # lost/found

    last_seen_lat: Mapped[float] = mapped_column(Float)
    last_seen_lng: Mapped[float] = mapped_column(Float)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    alert_radius_km: Mapped[float] = mapped_column(Float, default=16.0)
    alert_threshold: Mapped[float] = mapped_column(Float, default=0.60)

    # Contact details — NEVER exposed in public/match responses; revealed only
    # to a user with an active ContactUnlock for this pet.
    contact_name: Mapped[str] = mapped_column(String(120), default="")
    contact_email: Mapped[str] = mapped_column(String(160), default="")
    contact_phone: Mapped[str] = mapped_column(String(40), default="")

    photo_path: Mapped[str] = mapped_column(String(255), default="")
    photo_path2: Mapped[str] = mapped_column(String(255), default="")  # optional 2nd photo (display only)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)   # np.float32 bytes
    embed_model: Mapped[str] = mapped_column(String(60), default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Sighting(Base):
    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(primary_key=True)
    reporter_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    note: Mapped[str] = mapped_column(Text, default="")
    # Contact details of the finder — also gated behind an unlock.
    contact_name: Mapped[str] = mapped_column(String(120), default="")
    contact_email: Mapped[str] = mapped_column(String(160), default="")
    contact_phone: Mapped[str] = mapped_column(String(40), default="")

    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    search_radius_km: Mapped[float] = mapped_column(Float, default=16.0)
    status: Mapped[str] = mapped_column(String(20), default="open")

    photo_path: Mapped[str] = mapped_column(String(255), default="")
    photo_path2: Mapped[str] = mapped_column(String(255), default="")  # optional 2nd photo (display only)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    embed_model: Mapped[str] = mapped_column(String(60), default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class ContactUnlock(Base):
    """A paid 30-day pass: user X may see contact info for pet Y until expires_at."""
    __tablename__ = "contact_unlocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)

    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/active
    amount_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tip_usd: Mapped[float] = mapped_column(Float, default=0.0)
    # Reference the user provides after paying (Square receipt # or checkout email).
    payment_ref: Mapped[str] = mapped_column(String(200), default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    activated_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    def is_active(self) -> bool:
        return (
            self.status == "active"
            and self.expires_at is not None
            and self.expires_at > _now()
        )


class FinderUnlock(Base):
    """A paid 30-day PER-PET bundle in the OTHER direction: pet-owner X may see the
    finder contact for EVERY qualifying sighting of pet Y until expires_at.

    One payment ($24.99) unlocks all green+orange (>=65%) matches on that pet for
    30 days, including new qualifying sightings that arrive during the window.
    Keyed on pet_id (not sighting_id) so it covers many sightings at once.
    """
    __tablename__ = "finder_unlocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)

    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/active
    amount_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tip_usd: Mapped[float] = mapped_column(Float, default=0.0)
    payment_ref: Mapped[str] = mapped_column(String(200), default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    activated_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    def is_active(self) -> bool:
        return (
            self.status == "active"
            and self.expires_at is not None
            and self.expires_at > _now()
        )


class MatchAlert(Base):
    """Record that we emailed an alert for a given pet+sighting match.

    Used purely as a de-dup guard so an owner isn't emailed repeatedly about the
    same sighting matching the same pet. One row per (pet_id, sighting_id).
    """
    __tablename__ = "match_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)
    sighting_id: Mapped[int | None] = mapped_column(
        ForeignKey("sightings.id"), nullable=True, index=True
    )
    owner_emailed: Mapped[int] = mapped_column(Integer, default=0)   # 0/1
    finder_emailed: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    score_pct: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


def _ensure_columns() -> None:
    """Tiny idempotent migration: add photo_path2 to existing tables.

    create_all() only creates MISSING tables — it never alters existing ones.
    On an already-deployed DB (Render Postgres) the new photo_path2 columns
    won't exist, so add them here if missing. Safe to run every startup.
    """
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    for table in ("pets", "sightings"):
        if table not in insp.get_table_names():
            continue
        cols = {c["name"] for c in insp.get_columns(table)}
        if "photo_path2" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN photo_path2 VARCHAR(255) DEFAULT ''"
                ))
            print(f"[db] added photo_path2 to {table}", flush=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _ensure_columns()
