"""Vector store implementations.

Local-first design:
- Primary: Chroma PersistentClient
- Fallback: in-memory cosine similarity
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modules.embedder import Embedder
from modules.guideline_processor import GuidelineChunk


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    distance: float


class VectorStore:
    def index_guidelines(self, chunks: List[GuidelineChunk], embedder: Embedder) -> int:
        raise NotImplementedError

    def query(
        self,
        query_text: str,
        embedder: Embedder,
        top_k: int = 5,
        page_range: Optional[tuple[int, int]] = None,
    ) -> List[VectorSearchResult]:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        *,
        persist_path: Path,
        collection_name: str = "uganda_hiv_guidelines",
    ) -> None:
        self._persist_path = persist_path
        self._collection_name = collection_name

        # Rationale: ensure the directory exists before initializing the persistent client.
        self._persist_path.mkdir(parents=True, exist_ok=True)

        import chromadb

        # PersistentClient is the recommended way to persist locally. [src]
        # https://docs.trychroma.com/docs/run-chroma/persistent-client
        self._client = chromadb.PersistentClient(path=str(self._persist_path))

        # get_or_create_collection is documented for safe reuse. [src]
        # https://docs.trychroma.com/docs/collections/create-get-delete
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Uganda HIV guideline chunks"},
        )

    def index_guidelines(self, chunks: List[GuidelineChunk], embedder: Embedder) -> int:
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = embedder.encode(documents)
        metadatas = [
            {
                "page_number": c.page_number,
                "source_path": c.source_path,
            }
            for c in chunks
        ]

        # Use upsert to be idempotent across runs. [src]
        # https://docs.trychroma.com/docs/collections/update-data
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        return len(ids)

    def query(
        self,
        query_text: str,
        embedder: Embedder,
        top_k: int = 5,
        page_range: Optional[tuple[int, int]] = None,
    ) -> List[VectorSearchResult]:
        if not query_text.strip():
            return []

        query_embedding = embedder.encode([query_text])[0]

        # Query API documented here. [src]
        # https://docs.trychroma.com/docs/querying-collections/query-and-get
        # Rationale: Stage 2 support — optionally constrain by PDF page range using metadata filter.
        # Chroma 'where' filter with numeric ranges uses $gte/$lte. [src]
        # https://docs.trychroma.com/guides#using-where-to-filter-metadata
        where = None
        if page_range is not None:
            lo, hi = page_range
            # Chroma requires one operator per expression; use $and to combine range bounds.
            where = {
                "$and": [
                    {"page_number": {"$gte": int(lo)}},
                    {"page_number": {"$lte": int(hi)}},
                ]
            }

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
            where=where,
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        out: List[VectorSearchResult] = []
        for i in range(min(len(ids), len(documents), len(metadatas), len(distances))):
            out.append(
                VectorSearchResult(
                    chunk_id=str(ids[i]),
                    document=str(documents[i]),
                    metadata=dict(metadatas[i] or {}),
                    distance=float(distances[i]),
                )
            )
        return out


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._ids: List[str] = []
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None

    def index_guidelines(self, chunks: List[GuidelineChunk], embedder: Embedder) -> int:
        if not chunks:
            return 0

        self._ids = [c.chunk_id for c in chunks]
        self._docs = [c.text for c in chunks]
        self._metas = [
            {"page_number": c.page_number, "source_path": c.source_path}
            for c in chunks
        ]

        vectors = embedder.encode(self._docs)
        self._embeddings = np.asarray(vectors, dtype=np.float32)
        return len(self._ids)

    def query(
        self,
        query_text: str,
        embedder: Embedder,
        top_k: int = 5,
        page_range: Optional[tuple[int, int]] = None,
    ) -> List[VectorSearchResult]:
        if self._embeddings is None or not self._ids:
            return []
        if not query_text.strip():
            return []

        q = np.asarray(embedder.encode([query_text])[0], dtype=np.float32)

        # Cosine distance: 1 - cosine similarity.
        # Rationale: we use a standard similarity metric for dense embeddings.
        denom = (np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q))
        denom = np.where(denom == 0.0, 1e-12, denom)
        sims = (self._embeddings @ q) / denom
        distances = 1.0 - sims

        # Rationale: Stage 2 support — optionally restrict results to a PDF page range using stored metadata.
        if page_range is not None:
            lo, hi = page_range
            allowed = np.array(
                [
                    (m.get("page_number") is not None and int(m.get("page_number")) >= int(lo) and int(m.get("page_number")) <= int(hi))
                    for m in self._metas
                ],
                dtype=bool,
            )
            if not allowed.any():
                return []
            # Set distances of disallowed items to +inf so they cannot be selected in top_k.
            distances = distances.copy()
            distances[~allowed] = np.inf

        k = min(top_k, distances.shape[0])
        top_idx = np.argsort(distances)[:k]
        # Ensure we do not return items outside the allowed page range.
        if page_range is not None:
            top_idx = [int(i) for i in top_idx if allowed[int(i)]]
            if not top_idx:
                return []

        out: List[VectorSearchResult] = []
        for i in top_idx:
            out.append(
                VectorSearchResult(
                    chunk_id=self._ids[int(i)],
                    document=self._docs[int(i)],
                    metadata=dict(self._metas[int(i)]),
                    distance=float(distances[int(i)]),
                )
            )
        return out


def create_vector_store(
    *,
    project_root: Path,
    prefer_chroma: bool = True,
    collection_name: str = "uganda_hiv_guidelines",
) -> VectorStore:
    """Factory that returns the best available vector store implementation."""

    if prefer_chroma:
        try:
            # Rationale: separate collections per embedding model to avoid mixing
            # incompatible vector spaces in persistent storage.
            return ChromaVectorStore(
                persist_path=project_root / "storage" / "chroma",
                collection_name=collection_name,
            )
        except Exception:
            # Rationale: fallback keeps the demo runnable even if Chroma fails to initialize.
            return InMemoryVectorStore()

    return InMemoryVectorStore()
