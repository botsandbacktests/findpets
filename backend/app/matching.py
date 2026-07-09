"""Core matching: radius pre-filter + attribute filter + cosine similarity ranking."""
from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from .db import Pet
from .embedder import cosine_similarity
from .geo import haversine_km


def _bytes_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def find_matches(
    db: Session,
    query_vec: np.ndarray,
    query_model: str,
    lat: float,
    lng: float,
    radius_km: float,
    species: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Return ranked candidate lost pets for a sighting.

    Pipeline: filter active lost pets to those (a) within radius and
    (b) same embedding model, optionally (c) same species; then rank the
    survivors by cosine similarity of the DINOv2/fallback vectors.
    """
    pets = db.query(Pet).filter(Pet.status == "lost").all()
    results = []
    for pet in pets:
        if pet.embed_model != query_model:
            continue  # can't compare vectors from different backends
        dist = haversine_km(lat, lng, pet.last_seen_lat, pet.last_seen_lng)
        if dist > radius_km:
            continue
        if species and pet.species and species.lower() != pet.species.lower():
            continue
        sim = cosine_similarity(query_vec, _bytes_to_vec(pet.embedding))
        results.append({
            "pet": pet,
            "similarity": sim,
            "distance_km": round(dist, 2),
            "score_pct": round(sim * 100, 1),
        })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]
