"""Explanation generation.

This module produces clinician-facing text for each deterministic alert, using:
- retrieved guideline chunks (for traceable citations)
- local LLM (Ollama) when available

If the LLM is unavailable, it falls back to a deterministic explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Union
from typing import List, Optional

from modules.alert_rules import Alert
from modules.llm_client import ChatMessage, OllamaClient, HuggingFaceClient
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult

# Rationale: both OllamaClient and HuggingFaceClient have the same chat() interface.
LLMClient = Union[OllamaClient, HuggingFaceClient]


@dataclass(frozen=True)
class ExplanationResult:
    text: str
    used_chunks: List[VectorSearchResult]
    used_llm: bool


def generate_audit_checklist_alerts(
    *,
    patient_context: PatientContext,
    retrieved_chunks: List[VectorSearchResult],
    llm_client: Optional[LLMClient] = None,
) -> List[Alert]:
    """Generate visit audit checklist items as Alert objects.

    Notes:
    - This is an LLM-first layer intended to surface "things to check this visit".
    - Safe fallback: if retrieval is empty, return no checklist alerts.
    """

    used_chunks = list(retrieved_chunks[:5])
    if not used_chunks:
        return []

    if llm_client is None:
        return []

    llm_text = _try_llm_audit_checklist(
        llm_client=llm_client,
        patient_context=patient_context,
        retrieved_chunks=used_chunks,
    )
    if not llm_text:
        return []

    try:
        parsed = json.loads(llm_text)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    alerts: List[Alert] = []
    for i, item in enumerate(parsed[:6]):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "Guideline audit checklist item").strip()
        recommendation = str(item.get("recommendation") or "").strip()
        urgency = str(item.get("urgency") or "").strip()
        citations = item.get("citations")

        if not recommendation:
            continue

        msg_lines: List[str] = [recommendation]
        if urgency:
            msg_lines.append(f"Urgency/timeframe: {urgency}")
        message = "\n".join(msg_lines)

        evidence: Dict[str, Any] = {
            "type": "llm_audit_checklist",
            "item": item,
        }
        if citations is not None:
            evidence["citations"] = citations

        alerts.append(
            Alert(
                alert_id=f"llm_checklist_{i+1}",
                title=title,
                message=message,
                evidence=evidence,
                query_hint="visit audit checklist labs monitoring regimen",
            )
        )

    return alerts


def generate_explanation(
    *,
    patient_context: PatientContext,
    alert: Alert,
    retrieved_chunks: List[VectorSearchResult],
    llm_client: Optional[LLMClient] = None,
) -> ExplanationResult:
    """Generate an explanation for an alert."""

    used_chunks = list(retrieved_chunks[:3])

    # Rationale: safe fallback — if retrieval is empty, do not call the LLM.
    # This prevents guideline-sounding output without any retrieved evidence.
    if not used_chunks:
        return ExplanationResult(
            text=_fallback_explanation(patient_context, alert, used_chunks),
            used_chunks=used_chunks,
            used_llm=False,
        )

    if llm_client is not None:
        llm_text = _try_llm_explanation(
            llm_client=llm_client,
            patient_context=patient_context,
            alert=alert,
            retrieved_chunks=used_chunks,
        )
        if llm_text:
            # Rationale: guarantee at least one page citation appears when chunks exist.
            # We keep chunk_id traceability in the retrieved metadata, but avoid showing it
            # in clinician-facing text.
            if used_chunks and ("(page=" not in llm_text):
                citations_lines: List[str] = []
                for c in used_chunks:
                    page = c.metadata.get("page_number")
                    excerpt = (c.document or "")[:200]
                    citations_lines.append(f'- "{excerpt}" (page={page})')
                llm_text = llm_text + "\n\nCitations:\n" + "\n".join(citations_lines)
            return ExplanationResult(text=llm_text, used_chunks=used_chunks, used_llm=True)

    # Deterministic fallback.
    return ExplanationResult(
        text=_fallback_explanation(patient_context, alert, used_chunks),
        used_chunks=used_chunks,
        used_llm=False,
    )


def generate_stage3_synthesis_issues(
    *,
    patient_context: PatientContext,
    deterministic_alerts: List[Alert],
    evidence_map: Dict[str, List[VectorSearchResult]],
    llm_client: Optional[LLMClient],
) -> List[Alert]:
    """Single-call synthesis that produces structured issues and maps them to Alerts.

    Rationale: Stage 3 requires exactly one LLM call; we package a compact Stage 1
    alert summary and a Stage 2 evidence bundle (chunk_id, page_number, quote).
    The model returns JSON issues which we convert to `Alert` objects for gating.
    """

    if llm_client is None:
        return []

    # Build Stage 1 compact alert JSON.
    stage1 = [
        {
            "alert_id": a.alert_id,
            "title": a.title,
            "message": a.message,
            "evidence": a.evidence,
        }
        for a in deterministic_alerts
    ]

    # Build Stage 2 compact evidence bundle.
    stage2: Dict[str, List[Dict[str, Any]]] = {}
    for aid, items in evidence_map.items():
        bundle: List[Dict[str, Any]] = []
        for r in list(items or [])[:3]:
            bundle.append(
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.metadata.get("page_number"),
                    "quote": (r.document or "")[:320],
                }
            )
        stage2[aid] = bundle

    system: ChatMessage = {
        "role": "system",
        "content": (
            "You are a clinical decision support assistant. "
            "Use the provided alerts (Stage 1) and evidence bundle (Stage 2). "
            "Return ONLY valid JSON: a list of issues, each with keys: "
            "issue_type [monitoring_gap|ddi|toxicity_pattern], severity [high|medium|low], "
            "guideline_reference (section/page), already_in_plan [yes|no], nudge_needed [yes|no]."
        ),
    }

    user: ChatMessage = {
        "role": "user",
        "content": (
            f"Patient ID: {patient_context.patient_id}\n"
            f"Age: {patient_context.age_years if patient_context.age_years is not None else 'Not recorded'} years\n"
            f"Sex: {patient_context.sex if patient_context.sex else 'Not recorded'}\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n\n"
            f"Current ART regimen: {', '.join(patient_context.art_regimen_current) if patient_context.art_regimen_current else 'None'}\n"
            f"Other medications: {', '.join(patient_context.other_medications) if patient_context.other_medications else 'None'}\n\n"
            "Laboratory results:\n"
            f"{patient_context.labs_narrative if patient_context.labs_narrative else 'None recorded'}\n\n"
            "Stage 1 alerts (deterministic JSON):\n"
            f"{json.dumps(stage1, ensure_ascii=False)}\n\n"
            "Stage 2 evidence bundle (by alert_id):\n"
            f"{json.dumps(stage2, ensure_ascii=False)}\n\n"
            "Create a list of issues as specified and return ONLY JSON."
        ),
    }

    llm_text = llm_client.chat([system, user])
    if not llm_text:
        return []

    try:
        parsed = json.loads(llm_text)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    out: List[Alert] = []
    for i, item in enumerate(parsed[:8]):
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("issue_type") or "issue").strip()
        severity = str(item.get("severity") or "").strip()
        guideline_reference = str(item.get("guideline_reference") or "").strip()
        already_in_plan = str(item.get("already_in_plan") or "no").strip()
        nudge_needed = str(item.get("nudge_needed") or "no").strip()

        title = f"Synthesis: {issue_type}"
        if severity:
            title = f"{title} ({severity})"

        msg_lines = [f"Guideline reference: {guideline_reference}" if guideline_reference else "Proposed issue"]
        msg_lines.append(f"Already in plan: {already_in_plan}")
        msg_lines.append(f"Nudge needed: {nudge_needed}")
        message = "\n".join(msg_lines)

        evidence: Dict[str, Any] = {
            "type": "synthesis_issue",
            "issue": item,
        }

        out.append(
            Alert(
                alert_id=f"synth_issue_{i+1}",
                title=title,
                message=message,
                evidence=evidence,
                query_hint="stage3 synthesis issues",
            )
        )

    return out

def _try_llm_explanation(
    *,
    llm_client: LLMClient,
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
        excerpt = (c.document or "")[:600]
        # Rationale: keep citations readable by exposing page only (no chunk_id) in the
        # text the model sees; chunk_id remains available in VectorSearchResult metadata.
        chunks_block_lines.append(f"[page={page}] {excerpt}")

    chunks_block = "\n\n".join(chunks_block_lines) if chunks_block_lines else "(no chunks)"

    system: ChatMessage = {
        "role": "system",
        "content": (
            "You are a clinical assistant. Use only provided excerpts. Be concise. "
            "Return plain text with page citations (page=<page_number>). Brief quotes optional if they clarify the action."
        ),
    }

    user: ChatMessage = {
        "role": "user",
        "content": (
            f"Patient ID: {patient_context.patient_id}\n"
            f"Age: {patient_context.age_years if patient_context.age_years is not None else 'Not recorded'} years\n"
            f"Sex: {patient_context.sex if patient_context.sex else 'Not recorded'}\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n"
            f"Current ART regimen: {regimen}\n"
            f"Other medications: {', '.join(patient_context.other_medications) if patient_context.other_medications else 'None'}\n\n"
            f"Alert title: {alert.title}\n"
            f"Alert message: {alert.message}\n"
            f"Alert evidence: {alert.evidence}\n\n"
            "Guideline excerpts (use these for citations):\n"
            f"{chunks_block}\n\n"
            "Provide: (1) Clinical issue and (2) Recommended action. Include page citation."
        ),
    }

    return llm_client.chat([system, user])


def _try_llm_audit_checklist(
    *,
    llm_client: LLMClient,
    patient_context: PatientContext,
    retrieved_chunks: List[VectorSearchResult],
) -> Optional[str]:
    regimen = ", ".join(patient_context.art_regimen_current) or "unknown regimen"

    chunks_block_lines: List[str] = []
    for c in retrieved_chunks:
        page = c.metadata.get("page_number")
        excerpt = (c.document or "")[:600]
        chunks_block_lines.append(
            f"[chunk_id={c.chunk_id} page={page} distance={c.distance:.4f}] {excerpt}"
        )
    chunks_block = "\n\n".join(chunks_block_lines) if chunks_block_lines else "(no chunks)"

    system: ChatMessage = {
        "role": "system",
        "content": (
            "You are a clinical decision support assistant. "
            "Only use the provided guideline excerpts as evidence. "
            "If information is missing, say so. "
            "Return ONLY valid JSON (no markdown)."
        ),
    }

    user: ChatMessage = {
        "role": "user",
        "content": (
            f"Patient ID: {patient_context.patient_id}\n"
            f"Age: {patient_context.age_years if patient_context.age_years is not None else 'Not recorded'} years\n"
            f"Sex: {patient_context.sex if patient_context.sex else 'Not recorded'}\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n"
            f"Current ART regimen: {regimen}\n"
            f"Other medications: {', '.join(patient_context.other_medications) if patient_context.other_medications else 'None'}\n\n"
            f"Notes: {(patient_context.notes_text or '')}\n\n"
            f"Laboratory results:\n"
            f"{patient_context.labs_narrative if patient_context.labs_narrative else 'None recorded'}\n\n"
            "Guideline excerpts (use these for citations):\n"
            f"{chunks_block}\n\n"
            "Create a brief visit audit checklist (maximum 3-5 items): key monitoring points based on patient context and excerpts. "
            "Each item must cite page numbers. Brief rationale only. "
            "Output JSON as a list of objects with keys: title, recommendation, urgency, citations. "
            "The citations value must be a list of objects with keys: page_number, chunk_id. "
            "Return only JSON."
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
            # Rationale: clinician-facing fallback output should not include chunk_id.
            lines.append(f"- (page={page}) {excerpt}")
    else:
        lines.append("\nGuideline excerpts: none retrieved")

    return "\n".join(lines)
