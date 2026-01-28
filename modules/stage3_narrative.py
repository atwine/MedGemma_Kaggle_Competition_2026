from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from modules.llm_client import ChatMessage, OllamaClient
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult


def _build_stage2_bundle(evidence_map: Dict[str, List[VectorSearchResult]] ) -> Dict[str, List[Dict[str, Any]]]:
    bundle: Dict[str, List[Dict[str, Any]]] = {}
    for aid, items in (evidence_map or {}).items():
        small: List[Dict[str, Any]] = []
        for r in list(items or [])[:3]:
            small.append(
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.metadata.get("page_number"),
                    "quote": (r.document or "")[:320],
                }
            )
        bundle[aid] = small
    return bundle


def generate_stage3_narrative(
    *,
    patient_context: PatientContext,
    stage1_summary: Dict[str, Any],
    evidence_map: Dict[str, List[VectorSearchResult]],
    llm_client: Optional[OllamaClient],
) -> Optional[Dict[str, Any]]:
    """Generate a narrative JSON (Subjective/Objective/Assessment/Plan) without altering existing alerts.

    Returns None if llm_client is not available or output cannot be parsed as JSON.
    """

    if llm_client is None:
        return None

    stage2_bundle = _build_stage2_bundle(evidence_map)

    system: ChatMessage = {
        "role": "system",
        "content": (
            "You are a clinical decision support system for HIV medication safety. "
            "Use the provided Stage 1 summary (subjective+objective) and Stage 2 evidence bundle. "
            "Return ONLY valid JSON with top-level keys: Subjective, Objective, Assessment, Plan. "
            "For each item in Assessment, include fields: issue_type [monitoring_gap|ddi|toxicity_pattern], "
            "severity [high|medium|low], guideline_reference (section/page), already_in_plan [yes|no], nudge_needed [yes|no]."
        ),
    }

    user: ChatMessage = {
        "role": "user",
        "content": (
            f"Patient: {patient_context.name} ({patient_context.patient_id})\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n\n"
            "Stage 1 summary JSON:\n"
            f"{json.dumps(stage1_summary, ensure_ascii=False)}\n\n"
            "Stage 2 evidence bundle (by alert_id):\n"
            f"{json.dumps(stage2_bundle, ensure_ascii=False)}\n\n"
            "Compose the narrative strictly as JSON with keys: Subjective, Objective, Assessment, Plan."
        ),
    }

    llm_text = llm_client.chat([system, user])
    if not llm_text:
        return None

    # Minimal robustness: many chat LLMs wrap JSON in code fences or prepend notes.
    # Attempt direct parse first, then fall back to extracting the first JSON object span.
    try:
        parsed = json.loads(llm_text)
    except Exception:
        text = llm_text.strip()
        # Handle fenced blocks ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            # Remove first fence
            text = text.split("```", 1)[1]
            # If a language tag is present (e.g., json\n), strip up to first newline
            if "\n" in text:
                text = text.split("\n", 1)[1]
            # Trim trailing fence if present
            if "```" in text:
                text = text.rsplit("```", 1)[0]
        # Fallback: take the first {...} span
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                return None
        else:
            return None

    # Ensure the expected top-level shape exists
    out: Dict[str, Any] = {}
    out["Subjective"] = parsed.get("Subjective", [])
    out["Objective"] = parsed.get("Objective", [])
    out["Assessment"] = parsed.get("Assessment", [])
    out["Plan"] = parsed.get("Plan", [])
    return out
