"""RAG retrieval engine.

This module:
- builds a retrieval query from patient context + alert
- retrieves relevant guideline chunks from the local vector store
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from modules.alert_rules import Alert
from modules.embedder import Embedder
from modules.guideline_processor import GuidelineChunk, process_guidelines
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult, VectorStore


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 1200
    overlap: int = 200
    top_k: int = 5


class RagEngine:
    def __init__(
        self,
        *,
        project_root: Path,
        guideline_pdf_paths: List[Path],
        embedder: Embedder,
        vector_store: VectorStore,
        config: RagConfig | None = None,
    ) -> None:
        self._project_root = project_root
        self._guideline_pdf_paths = list(guideline_pdf_paths)
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config or RagConfig()

        self._indexed = False

    def ensure_indexed(self, *, max_pages: Optional[int] = None) -> int:
        """Index guideline chunks into the vector store.

        Notes:
        - Indexing is cached per-process using `_indexed` for demo simplicity.
        - Uses `upsert` under the hood (Chroma) to keep re-runs idempotent.
        """

        if self._indexed:
            return 0

        total = 0
        for pdf_path in self._guideline_pdf_paths:
            chunks: List[GuidelineChunk] = process_guidelines(
                pdf_path,
                chunk_size=self._config.chunk_size,
                overlap=self._config.overlap,
                max_pages=max_pages,
            )

            doc_prefix = (
                pdf_path.stem.strip().lower().replace(" ", "_").replace("+", "_")
            )
            prefixed = [
                GuidelineChunk(
                    chunk_id=f"{doc_prefix}__{c.chunk_id}",
                    text=c.text,
                    source_path=c.source_path,
                    page_number=c.page_number,
                )
                for c in chunks
            ]
            total += self._vector_store.index_guidelines(prefixed, self._embedder)
        self._indexed = True
        return total

    def retrieve_for_alert(
        self,
        *,
        patient_context: PatientContext,
        alert: Alert,
        top_k: Optional[int] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> List[VectorSearchResult]:
        """Retrieve guideline chunks relevant to a given alert."""

        query = self._build_query(patient_context=patient_context, alert=alert)
        k = self._config.top_k if top_k is None else top_k
        # Rationale: Stage 2 support — optionally restrict retrieval to a PDF page range.
        results = self._vector_store.query(query, self._embedder, top_k=k, page_range=page_range)
        # Defensive: enforce the page bounds even if the underlying store ignores the filter.
        if page_range is not None:
            lo, hi = page_range
            results = [
                r for r in results
                if (r.metadata.get("page_number") is not None and int(r.metadata.get("page_number")) >= int(lo) and int(r.metadata.get("page_number")) <= int(hi))
            ]
        return results

    def _build_query(self, *, patient_context: PatientContext, alert: Alert) -> str:
        # Rationale: We keep query building simple and transparent for auditability.
        regimen = ", ".join(patient_context.art_regimen_current) or "unknown regimen"
        note_excerpt = (patient_context.notes_text or "").strip()[:500]

        # Rationale: include alert details to improve retrieval specificity; the LLM
        # still only sees the top_k retrieved excerpts, not the whole PDF.
        evidence_excerpt = str(alert.evidence or "")[:400]
        alert_message_excerpt = (alert.message or "").strip()[:300]

        return (
            f"Alert: {alert.title}\n"
            f"Hint: {alert.query_hint}\n"
            f"Alert message: {alert_message_excerpt}\n"
            f"Alert evidence: {evidence_excerpt}\n"
            f"Patient regimen: {regimen}\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n"
            f"Notes excerpt: {note_excerpt}"
        )
