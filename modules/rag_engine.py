"""RAG retrieval engine.

This module:
- builds a retrieval query from patient context + alert
- retrieves relevant guideline chunks from the local vector store
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import re

from modules.alert_rules import Alert
from modules.embedder import Embedder
from modules.guideline_processor import GuidelineChunk, process_guidelines, process_markdown_guidelines
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
        guideline_paths: List[Path] | None = None,
        guideline_pdf_paths: List[Path] | None = None,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RagConfig | None = None,
    ) -> None:
        self._project_root = project_root
        # Rationale: accept both new 'guideline_paths' and legacy 'guideline_pdf_paths'
        # to avoid breaking existing call-sites.
        self._guideline_paths = list(guideline_paths or guideline_pdf_paths or [])
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
        for gpath in self._guideline_paths:
            # Rationale: detect Markdown vs PDF by extension so callers can mix
            # file types in a single guideline list.
            if gpath.suffix.lower() in (".md", ".markdown"):
                chunks: List[GuidelineChunk] = process_markdown_guidelines(
                    gpath,
                    chunk_size=self._config.chunk_size,
                    overlap=self._config.overlap,
                    max_pages=max_pages,
                )
            else:
                chunks = process_guidelines(
                    gpath,
                    chunk_size=self._config.chunk_size,
                    overlap=self._config.overlap,
                    max_pages=max_pages,
                )

            doc_prefix = (
                gpath.stem.strip().lower().replace(" ", "_").replace("+", "_")
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
        notes_text = (patient_context.notes_text or "").strip()
        note_excerpt = notes_text[:500]

        # Rationale: manual cases may only contain key facts (e.g., TDF, phosphate, DEXA)
        # later in a long history note; extract a small keyword set from the full text
        # so retrieval is not dominated by the first 500 characters.
        keywords = _extract_retrieval_keywords(notes_text)

        # Rationale: include alert details to improve retrieval specificity; the LLM
        # still only sees the top_k retrieved excerpts, not the whole PDF.
        evidence_excerpt = str(alert.evidence or "")[:400]
        alert_message_excerpt = (alert.message or "").strip()[:300]

        return (
            f"Alert: {alert.title}\n"
            f"Hint: {alert.query_hint}\n"
            f"Alert message: {alert_message_excerpt}\n"
            f"Alert evidence: {evidence_excerpt}\n"
            f"Patient keywords: {', '.join(keywords) if keywords else 'none'}\n"
            f"Patient regimen: {regimen}\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n"
            f"Notes excerpt: {note_excerpt}"
        )


def _extract_retrieval_keywords(notes_text: str) -> List[str]:
    # Rationale: keep this deterministic and small; this is used only to improve
    # retrieval recall for history-only free-text cases.
    if not notes_text:
        return []

    text = notes_text.upper()

    # Rationale: intentionally small vocabulary — expand only when justified.
    arv_terms = {
        "TDF",
        "3TC",
        "DTG",
        "AZT",
        "FTC",
        "EFV",
        "NVP",
        "D4T",
        "ABC",
        "TAF",
    }
    clinical_terms = {
        "BONE PAIN",
        "FRACTURE",
        "OSTEOPOROSIS",
        "DEXA",
        "T-SCORE",
        "LOOSER",
        "PHOSPHATE",
        "URINALYSIS",
        "PROTEIN",
        "GLUCOSE",
        "CREATININE",
        "EGFR",
        "VIRAL LOAD",
        "CD4",
        "VITAMIN D",
    }

    found: List[str] = []
    for term in sorted(arv_terms | clinical_terms):
        if term in text:
            found.append(term)

    # Rationale: also capture common numeric-lab patterns present in text blocks
    # (e.g., "PHOSPHATE: 1.6", "T-score = -2.8").
    for m in re.finditer(r"\b(T-SCORE|PHOSPHATE|CREATININE|EGFR|CD4|VIRAL LOAD)\b[^\n]{0,60}", text):
        snippet = m.group(0).strip()
        if snippet and snippet not in found:
            found.append(snippet)

    return found[:30]
