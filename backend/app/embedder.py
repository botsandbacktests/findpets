"""
Image embedding for visual pet matching.

Three interchangeable backends behind one interface:

  1. HostedEmbedder  — REAL CLIP embeddings from a hosted API (Replicate). No torch,
                       no multi-GB install, runs fine on a 512MB Render box. This is
                       the production path: strong fine-grained similarity, pay-per-use.
  2. DINOv2Embedder  — real Meta DINOv2 ViT features (requires torch + torchvision).
                       Best local option, but too heavy for Render's free tier.
  3. FallbackEmbedder — lightweight, dependency-free descriptor built from color
                       histograms + edge/texture stats. Deterministic, always works,
                       lets the whole app run without any ML at all. Weak matching —
                       only a safety net when no API key and no torch are available.

All three return an L2-normalized float32 vector, so cosine similarity == dot product,
and all are swappable without any schema or API changes (the DB just stores a vector).

get_embedder() picks, in order: HostedEmbedder if REPLICATE_API_TOKEN is set,
then DINOv2 if torch is importable, else the fallback.
"""
from __future__ import annotations

import io
import os
import time
import json
import urllib.request
import urllib.error
import base64
import numpy as np
from PIL import Image, ImageFilter


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


class BaseEmbedder:
    name = "base"
    dim = 0

    def embed(self, image_bytes: bytes) -> np.ndarray:
        raise NotImplementedError


class FallbackEmbedder(BaseEmbedder):
    """
    Dependency-free descriptor. Not as strong as DINOv2, but captures coat color,
    brightness distribution, and coarse edge/texture structure — enough to
    demonstrate ranked visual matching end to end.
    """
    name = "fallback-colortexture"
    # 3 color channels * 16 bins + 16 grayscale bins + 8 edge-orientation bins
    dim = 3 * 16 + 16 + 8

    def embed(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((160, 160))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HxWx3

        feats = []
        # Per-channel color histograms
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1))
            feats.append(hist)

        # Grayscale intensity histogram
        gray = arr.mean(axis=2)
        ghist, _ = np.histogram(gray, bins=16, range=(0, 1))
        feats.append(ghist)

        # Coarse edge-orientation histogram (texture / shape cue)
        gimg = Image.fromarray((gray * 255).astype(np.uint8))
        gx = np.asarray(gimg.filter(ImageFilter.Kernel((3, 3),
              [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)), dtype=np.float32)
        gy = np.asarray(gimg.filter(ImageFilter.Kernel((3, 3),
              [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1)), dtype=np.float32)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = (np.arctan2(gy, gx) + np.pi)  # 0..2pi
        ohist, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi), weights=mag)
        feats.append(ohist)

        vec = np.concatenate(feats)
        return _l2_normalize(vec)


class HostedEmbedder(BaseEmbedder):
    """
    REAL CLIP image embeddings via the Replicate hosted API.

    Sends the image to Replicate's `krthr/clip-embeddings` model (CLIP
    ViT-L/14, 768-dim) and gets back a semantic feature vector. No torch, no
    model download — just an HTTPS call — so it runs on Render's free tier.

    Needs env var REPLICATE_API_TOKEN. Optionally REPLICATE_CLIP_VERSION to pin
    a specific model version hash.
    """
    name = "clip-vit-l14-hosted"
    dim = 768

    # Pinned version of krthr/clip-embeddings (CLIP ViT-L/14, 768-dim output).
    _DEFAULT_VERSION = (
        "1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4"
    )
    _API_URL = "https://api.replicate.com/v1/predictions"

    def __init__(self):
        self._token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
        if not self._token:
            raise RuntimeError("REPLICATE_API_TOKEN not set")
        self._version = os.environ.get(
            "REPLICATE_CLIP_VERSION", self._DEFAULT_VERSION
        ).strip()

    def _post(self, url: str, payload: dict | None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {self._token}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, image_bytes: bytes) -> np.ndarray:
        # Send the image inline as a data URI so we never depend on a public URL.
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"

        payload = {"version": self._version, "input": {"image": data_uri}}
        pred = self._post(self._API_URL, payload)

        # Poll until the prediction finishes (CLIP embeds run in a few seconds).
        get_url = pred.get("urls", {}).get("get")
        deadline = time.time() + 90
        while pred.get("status") not in ("succeeded", "failed", "canceled"):
            if time.time() > deadline:
                raise TimeoutError("Replicate embedding timed out")
            time.sleep(1.0)
            pred = self._get(get_url)

        if pred.get("status") != "succeeded":
            raise RuntimeError(f"Replicate embedding {pred.get('status')}: "
                               f"{pred.get('error')}")

        out = pred.get("output")
        # Model returns {"embedding": [...]} or a bare list depending on version.
        if isinstance(out, dict):
            out = out.get("embedding") or out.get("embeddings")
        vec = np.asarray(out, dtype=np.float32).ravel()
        if vec.size == 0:
            raise RuntimeError("Replicate returned empty embedding")
        return _l2_normalize(vec)


class DINOv2Embedder(BaseEmbedder):
    """Real DINOv2 ViT-S/14 embeddings via torch.hub. ~384-dim CLS token."""
    name = "dinov2-vits14"
    dim = 384

    def __init__(self):
        import torch
        import torchvision.transforms as T

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Loaded once, cached by torch.hub.
        self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self._model.eval().to(self._device)
        self._tf = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def embed(self, image_bytes: bytes) -> np.ndarray:
        torch = self._torch
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self._tf(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(x)  # (1, 384)
        vec = feat.squeeze(0).cpu().numpy()
        return _l2_normalize(vec)


_EMBEDDER: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """
    Return a cached embedder. Preference order:
      1. HostedEmbedder  (real CLIP via API)  — when REPLICATE_API_TOKEN is set.
      2. DINOv2Embedder  (real, local)        — when torch is importable.
      3. FallbackEmbedder (weak color/texture) — last resort so the app still runs.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    # 1. Hosted CLIP API — the production path (works on Render's free tier).
    if os.environ.get("REPLICATE_API_TOKEN", "").strip():
        try:
            _EMBEDDER = HostedEmbedder()
            print(f"[embedder] Using REAL hosted CLIP "
                  f"({_EMBEDDER.name}, dim={_EMBEDDER.dim})")
            return _EMBEDDER
        except Exception as e:
            print(f"[embedder] hosted API unavailable ({type(e).__name__}: {e}); "
                  f"trying next backend")

    # 2. Local DINOv2 (needs torch — too heavy for Render free tier, fine locally).
    try:
        import torch  # noqa: F401
        _EMBEDDER = DINOv2Embedder()
        print(f"[embedder] Using REAL DINOv2 ({_EMBEDDER.name}, dim={_EMBEDDER.dim})")
        return _EMBEDDER
    except Exception as e:  # torch missing or model load failed
        pass

    # 3. Last resort: weak, dependency-free descriptor.
    _EMBEDDER = FallbackEmbedder()
    print(f"[embedder] WARNING: no real model available; "
          f"using WEAK fallback embedder ({_EMBEDDER.name}, dim={_EMBEDDER.dim})")
    return _EMBEDDER


_FALLBACK_EMBEDDER: FallbackEmbedder | None = None


def embed_with_fallback(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Embed an image, degrading gracefully instead of crashing.

    Tries the preferred embedder (hosted CLIP / DINOv2). If it raises for ANY
    reason — e.g. Replicate returns HTTP 402 Payment Required, a network blip,
    or a timeout — we fall back to the always-available FallbackEmbedder so the
    user's action (posting a lost pet, reporting a sighting) still succeeds.

    Returns (vector, model_name). The model_name tells the caller which backend
    actually produced the vector, so it gets stored on the row and matching only
    ever compares vectors from the same backend.
    """
    global _FALLBACK_EMBEDDER
    emb = get_embedder()
    try:
        return emb.embed(image_bytes), emb.name
    except Exception as e:
        # If the primary WAS already the fallback, there's nothing better to try.
        if isinstance(emb, FallbackEmbedder):
            raise
        print(f"[embedder] primary '{emb.name}' failed "
              f"({type(e).__name__}: {e}); using fallback for this image",
              flush=True)
        if _FALLBACK_EMBEDDER is None:
            _FALLBACK_EMBEDDER = FallbackEmbedder()
        return _FALLBACK_EMBEDDER.embed(image_bytes), _FALLBACK_EMBEDDER.name


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-L2-normalized vectors of the same backend."""
    if a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))
# end of embedder module
