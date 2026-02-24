"""Embedding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import hashlib
import math

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import-time failures on offline envs
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class EmbedderConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None


class Embedder:
    """Light wrapper around SentenceTransformers for consistent embedding output."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self._config = config or EmbedderConfig()
        self._model: Optional[SentenceTransformer] = None  # type: ignore[assignment]
        # Rationale: try to load the transformer; if it fails (e.g., offline), fall back to a light hashing embedder.
        try:
            if SentenceTransformer is not None:
                self._model = SentenceTransformer(
                    self._config.model_name,
                    device=self._config.device,
                )
        except Exception:
            self._model = None
        # Fallback vector size (kept modest for laptop-friendly performance)
        self._fallback_dim = 256

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into dense vectors.

        Uses `SentenceTransformer.encode()` as shown in the official docs. [src]
        - https://www.sbert.net/docs/sentence_transformer/usage/usage.html
        """

        if not texts:
            return []

        if self._model is not None:
            # Note: `SentenceTransformer.encode()` returns a numpy array by default.
            embeddings = self._model.encode(texts)
            return embeddings.tolist()

        # Offline fallback: simple hashing-based bag-of-words embedding.
        out: List[List[float]] = []
        dim = self._fallback_dim
        for t in texts:
            vec = [0.0] * dim
            for token in str(t or "").lower().split():
                h = hashlib.sha256(token.encode("utf-8")).digest()
                # Use first 4 bytes to pick an index deterministically
                idx = int.from_bytes(h[:4], "big") % dim
                vec[idx] += 1.0
            # L2 normalize to approximate cosine behavior
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
