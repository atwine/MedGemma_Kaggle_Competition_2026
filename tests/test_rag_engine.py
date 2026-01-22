from __future__ import annotations

from dataclasses import dataclass
from typing import List

from modules.guideline_processor import GuidelineChunk
from modules.vector_store import InMemoryVectorStore


@dataclass
class _DummyEmbedder:
    # Rationale: avoid downloading external embedding models during tests.
    def encode(self, texts: List[str]) -> List[List[float]]:
        # Deterministic, low-dimensional embedding based on simple string features.
        out: List[List[float]] = []
        for t in texts:
            s = t or ""
            out.append([float(len(s)), float(s.count("a")), float(s.count("e"))])
        return out


def test_vector_retrieval_returns_top_k() -> None:
    store = InMemoryVectorStore()
    embedder = _DummyEmbedder()

    chunks = [
        GuidelineChunk(chunk_id="c1", text="alpha", source_path="/tmp/g.pdf", page_number=1),
        GuidelineChunk(chunk_id="c2", text="beta", source_path="/tmp/g.pdf", page_number=2),
        GuidelineChunk(chunk_id="c3", text="gamma", source_path="/tmp/g.pdf", page_number=3),
        GuidelineChunk(chunk_id="c4", text="delta", source_path="/tmp/g.pdf", page_number=4),
    ]

    store.index_guidelines(chunks, embedder)  # type: ignore[arg-type]

    results = store.query("alpha", embedder, top_k=3)  # type: ignore[arg-type]

    assert len(results) == 3
    assert all(r.chunk_id for r in results)
