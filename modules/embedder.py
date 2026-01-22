"""Embedding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer


@dataclass
class EmbedderConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None


class Embedder:
    """Light wrapper around SentenceTransformers for consistent embedding output."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self._config = config or EmbedderConfig()
        # Rationale: we keep a single model instance to avoid repeated load latency.
        self._model = SentenceTransformer(
            self._config.model_name,
            device=self._config.device,
        )

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

        # Note: `SentenceTransformer.encode()` returns a numpy array by default.
        embeddings = self._model.encode(texts)
        return embeddings.tolist()
