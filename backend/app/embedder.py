"""
Image embedding for visual pet matching.

Two interchangeable backends behind one interface:

  1. DINOv2Embedder  — real Meta DINOv2 ViT features (requires torch + torchvision).
                       Best fine-grained visual similarity for pet re-identification.
  2. FallbackEmbedder — lightweight, dependency-free descriptor built from color
                       histograms + edge/texture stats. Deterministic, always works,
                       lets the whole app run without a multi-GB ML install.

Both return an L2-normalized float32 vector, so cosine similarity == dot product,
and both are swappable without any schema or API changes (the DB just stores a vector).

get_embedder() picks DINOv2 if torch is importable, else the fallback.
"""
from __future__ import annotations

import io
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
    """Return a cached embedder, preferring real DINOv2 when torch is available."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        import torch  # noqa: F401
        _EMBEDDER = DINOv2Embedder()
        print(f"[embedder] Using REAL DINOv2 ({_EMBEDDER.name}, dim={_EMBEDDER.dim})")
    except Exception as e:  # torch missing or model load failed
        _EMBEDDER = FallbackEmbedder()
        print(f"[embedder] torch unavailable ({type(e).__name__}); "
              f"using fallback embedder ({_EMBEDDER.name}, dim={_EMBEDDER.dim})")
    return _EMBEDDER


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-L2-normalized vectors of the same backend."""
    if a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))
