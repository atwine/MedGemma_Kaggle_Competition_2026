"""Explanation generation.

This module produces clinician-facing text for each deterministic alert, using:
- retrieved guideline chunks (for traceable citations)
- local LLM (Ollama) when available

If the LLM is unavailable, it falls back to a deterministic explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from modules.alert_rules import Alert
from modules.llm_client import ChatMessage, OllamaClient
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult


@dataclass(frozen=True)
class ExplanationResult:
    text: str
    used_chunks: List[VectorSearchResult]
    used_llm: bool


def generate_explanation(
    *,
    patient_context: PatientContext,
    alert: Alert,
    retrieved_chunks: List[VectorSearchResult],
    llm_client: Optional[OllamaClient] = None,
) -> ExplanationResult:
    """Generate an explanation for an alert."""

    used_chunks = list(retrieved_chunks[:3])

    if llm_client is not None:
        llm_text = _try_llm_explanation(
            llm_client=llm_client,
            patient_context=patient_context,
            alert=alert,
            retrieved_chunks=used_chunks,
        )
        if llm_text:
            return ExplanationResult(text=llm_text, used_chunks=used_chunks, used_llm=True)

    # Deterministic fallback.
    return ExplanationResult(
        text=_fallback_explanation(patient_context, alert, used_chunks),
        used_chunks=used_chunks,
        used_llm=False,
    )


def _try_llm_explanation(
    *,
    llm_client: OllamaClient,
    patient_context: PatientContext,
    alert: Alert,
    retrieved_chunks: List[VectorSearchResult],
) -> Optional[str]:
    # Rationale: ensure the model response stays grounded in retrieved evidence.
    # We provide the chunks and explicitly require citations to the provided excerpts.

    regimen = ", ".join(patient_context.art_regimen_current) or "unknown regimen"

    chunks_block_lines: List[str] = []
    for c in retrieved_chunks:
        page = c.metadata.get("page_number")
        chunks_block_lines.append(
            f"[chunk_id={c.chunk_id} page={page} distance={c.distance:.4f}] {c.document}"
        )

    chunks_block = "\n\n".join(chunks_block_lines) if chunks_block_lines else "(no chunks)"

    system: ChatMessage = {
        "role": "system",
        "content": (
            "You are a clinical decision support assistant. "
            "Only use the provided guideline excerpts as evidence. "
            "If information is missing, say so. "
            "Always include at least one *quoted* guideline excerpt and a citation in the form (page=<page_number>, chunk_id=<chunk_id>). "
            "Use suggestive language (e.g., 'Consider...')."
        ),
    }

    user: ChatMessage = {
        "role": "user",
        "content": (
            f"Patient: {patient_context.name} ({patient_context.patient_id})\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n"
            f"Current ART regimen: {regimen}\n\n"
            f"Alert title: {alert.title}\n"
            f"Alert message: {alert.message}\n"
            f"Alert evidence: {alert.evidence}\n\n"
            "Guideline excerpts (use these for citations):\n"
            f"{chunks_block}\n\n"
            "Write a short explanation with:\n"
            "1) Why this alert triggered (based on evidence)\n"
            "2) What the clinician might consider doing next\n"
            "3) One or more citations to the provided excerpts"
        ),
    }

    return llm_client.chat([system, user])


def _fallback_explanation(
    patient_context: PatientContext,
    alert: Alert,
    retrieved_chunks: List[VectorSearchResult],
) -> str:
    # Rationale: deterministic fallback ensures the app is usable without the LLM.
    lines: List[str] = []
    lines.append(f"Why this alert: {alert.message}")
    lines.append(f"Patient evidence: {alert.evidence}")

    if retrieved_chunks:
        lines.append("\nGuideline excerpts:")
        for c in retrieved_chunks:
            page = c.metadata.get("page_number")
            excerpt = (c.document or "")[:350]
            lines.append(f"- (page={page}, chunk_id={c.chunk_id}) {excerpt}")
    else:
        lines.append("\nGuideline excerpts: none retrieved")

    return "\n".join(lines)
