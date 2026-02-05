"""Agentic RAG orchestration skeleton (Phase 1).

This module defines the core data structures and a minimal orchestrator for a
"light decomposition" agentic RAG flow. The current implementation is a
non-invasive skeleton:

- it does not call the embedder, RagEngine, or LLM client yet
- it is not wired into the existing analysis pipeline
- it is safe to import and call, but only returns placeholder results

Later phases can extend the stubbed steps to integrate planning, per-sub-task
retrieval, reasoning, and verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from modules.embedder import Embedder
from modules.llm_client import OllamaClient
from modules.patient_parser import build_patient_context
from modules.vector_store import VectorStore, VectorSearchResult


# Core data structures -----------------------------------------------------


@dataclass(frozen=True)
class QueryContext:
    """Normalized view of a user query and associated patient data.

    This is the primary object passed between agentic steps.
    """

    question_text: str
    patient_raw: Dict[str, Any]
    patient_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubTaskPlan:
    """One planning unit produced by the Planner agent.

    Phase 1 uses this as a structural placeholder; fields will be populated by a
    real planning step in Phase 2.
    """

    name: str
    description: str
    retrieval_query: str
    priority: str = "medium"


@dataclass
class EvidenceBundle:
    """Container for evidence retrieved for a given sub-task.

    In later phases, ``chunks`` will be populated from the vector store
    retrieval; for now this remains an empty placeholder.
    """

    subtask_name: str
    chunks: List[Any] = field(default_factory=list)
    summary: Optional[str] = None


@dataclass
class AgenticResult:
    """Final result of an agentic flow run.

    ``debug_info`` is intended for logs or future debug UI and may contain any
    intermediate artefacts (plans, evidence, verifier reports, etc.).
    """

    final_answer_text: str
    used_chunks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


# Stubbed agentic steps ----------------------------------------------------


def _normalize_input(context: QueryContext) -> QueryContext:
    """Phase 1 input normalization stub.

    Later phases may:
    - derive a structured patient summary from ``patient_raw``
    - enrich ``metadata`` with additional routing information

    For now, the input context is returned unchanged to keep behaviour explicit
    and deterministic.
    """

    return context


def _plan_subtasks(context: QueryContext) -> List[SubTaskPlan]:
    """Deterministic planner for light decomposition.

    Phase 2 introduces a minimal, code-based planner (no LLM calls yet) that
    always returns a small set of sub-tasks. This keeps behaviour predictable
    while allowing per-sub-task retrieval to be exercised in isolation.
    """

    question = (context.question_text or "").strip()
    if not question:
        question = "Clinical guideline support for this patient."

    subtasks: List[SubTaskPlan] = [
        SubTaskPlan(
            name="primary_question",
            description=(
                "Retrieve guideline sections that address the main clinical "
                "question for this patient."
            ),
            retrieval_query=question,
            priority="high",
        ),
        SubTaskPlan(
            name="safety_and_monitoring",
            description=(
                "Retrieve guideline sections about contraindications, "
                "cautions, monitoring, and follow-up for this scenario."
            ),
            retrieval_query=(
                f"{question}\n\n"
                "Focus on contraindications, cautions, monitoring, and follow-up."
            ),
            priority="medium",
        ),
    ]

    return subtasks


def _retrieve_evidence(
    context: QueryContext,
    subtasks: List[SubTaskPlan],
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    top_k: int = 5,
    page_range: Optional[Tuple[int, int]] = None,
) -> List[EvidenceBundle]:
    """Retrieve evidence for each sub-task using the existing vector store.

    This function reuses the established embedding + vector-store stack
    (``Embedder`` + ``VectorStore``) and constructs a query per sub-task that
    incorporates the user question and, when possible, a brief patient
    context derived from ``patient_raw``.
    """

    if not subtasks:
        return []

    # Build a lightweight patient context for query enrichment; fall back to
    # question-only retrieval if parsing fails or patient data is incomplete.
    patient_notes: str = ""
    regimen: str = ""
    encounter_date: str = ""
    try:
        patient_ctx = build_patient_context(context.patient_raw)
        regimen = ", ".join(patient_ctx.art_regimen_current) or "unknown regimen"
        patient_notes = (patient_ctx.notes_text or "").strip()
        encounter_date = patient_ctx.encounter_date.isoformat()
    except Exception:
        patient_ctx = None  # type: ignore[assignment]

    notes_excerpt = patient_notes[:500] if patient_notes else ""

    bundles: List[EvidenceBundle] = []

    for subtask in subtasks:
        parts: List[str] = []
        if context.question_text:
            parts.append(f"Question: {context.question_text.strip()}")
        if regimen or encounter_date:
            parts.append(
                f"Patient regimen: {regimen or 'unknown'}; "
                f"Encounter date: {encounter_date or 'unknown'}"
            )
        if notes_excerpt:
            parts.append(f"Patient notes excerpt: {notes_excerpt}")

        # Include the sub-task description and retrieval hint so the query text
        # remains transparent and auditable.
        parts.append(f"Subtask: {subtask.name}")
        parts.append(f"Subtask description: {subtask.description}")
        if subtask.retrieval_query:
            parts.append(f"Subtask retrieval focus: {subtask.retrieval_query}")

        query_text = "\n".join(parts).strip()
        if not query_text:
            continue

        results: List[VectorSearchResult] = vector_store.query(
            query_text=query_text,
            embedder=embedder,
            top_k=top_k,
            page_range=page_range,
        )

        bundles.append(EvidenceBundle(subtask_name=subtask.name, chunks=list(results)))

    return bundles


def _reason(
    context: QueryContext,
    subtasks: List[SubTaskPlan],
    evidence_bundles: List[EvidenceBundle],
    llm_client: Optional[OllamaClient] = None,
) -> str:
    """Reason over retrieved evidence.

    When ``llm_client`` is provided, this uses the same LLM pattern as
    ``agentic_demo.py`` to generate a grounded answer from the retrieved
    guideline excerpts. When no client is available, it falls back to the
    original placeholder text to keep behaviour explicit.
    """

    if llm_client is None:
        # Keep the original clearly-marked placeholder when no LLM is wired in.
        _ = (subtasks, evidence_bundles)  # explicit: unused in this branch
        return (
            "Agentic flow skeleton is not yet implemented. "
            "Question received but no agentic reasoning has been run."
        )

    # LLM-based reasoning path (debug-only; callers must ensure this is gated).
    try:
        evidence_lines: List[str] = []
        for bundle in evidence_bundles:
            chunks = list(bundle.chunks or [])
            if not chunks:
                continue
            evidence_lines.append(f"Subtask {bundle.subtask_name}:")
            for r in chunks[:2]:
                try:
                    metadata = getattr(r, "metadata", {}) or {}
                    page = metadata.get("page_number")
                    chunk_id = getattr(r, "chunk_id", None)
                    doc_text = getattr(r, "document", "") or ""
                except Exception:
                    continue
                preview = doc_text.replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                evidence_lines.append(
                    f"- page {page}, chunk {chunk_id}: {preview}"
                )
            evidence_lines.append("")

        evidence_text = "\n".join(evidence_lines) or "No evidence retrieved."

        patient = context.patient_raw or {}
        question = (context.question_text or "").strip() or (
            "Clinical guideline support for this patient."
        )
        patient_id = patient.get("patient_id")
        patient_name = patient.get("name")

        # Doctor-style rubric: nudge the model to reason like a clinician
        # (identify trends, consider toxicity and causes, map to guideline
        # actions, and propose a concrete assessment and plan) while still
        # staying strictly grounded in the retrieved excerpts.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical guideline assistant helping an HIV "
                    "clinician. Think step-by-step like a careful doctor. "
                    "First, build a concise problem list and note important "
                    "trends over time (for example creatinine/eGFR, viral "
                    "load, CD4, weight, key symptoms). Next, for each problem "
                    "consider possible causes, especially medication toxicity "
                    "from current or past ART or common co-medications such as "
                    "NSAIDs, as well as relevant comorbidities. Then, using "
                    "ONLY the provided guideline excerpts, determine how the "
                    "guidelines classify or stage the problem and what actions "
                    "they recommend (such as when to modify ART, avoid specific "
                    "drugs, or change doses). Finally, write an Assessment and "
                    "Plan that clearly links each problem to the most likely "
                    "cause and to explicit guideline-based actions, including "
                    "any counselling and recommended monitoring or follow-up. "
                    "If the excerpts are insufficient or do not cover the case, "
                    "say that you are unsure instead of guessing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient id: {patient_id}, name: {patient_name}\n"
                    f"Question: {question}\n\n"
                    f"Guideline excerpts:\n{evidence_text}"
                ),
            },
        ]

        answer = llm_client.chat(messages)
        if answer is None:
            # Degrade gracefully to a descriptive fallback while surfacing any
            # available error detail from the client.
            base = (
                "Agentic reasoning attempted but the LLM call did not return "
                "content. See Ollama logs for details."
            )
            last_err = getattr(llm_client, "last_error", None)
            if last_err:
                return f"{base} Last error: {last_err}"
            return base

        return answer
    except Exception:
        # Defensive: never break callers because of agentic reasoning failures.
        return (
            "Agentic flow skeleton is not yet implemented. "
            "Question received but no agentic reasoning has been run."
        )


def _verify(
    draft_answer: str,
    evidence_bundles: List[EvidenceBundle],
    context: QueryContext,
) -> Dict[str, Any]:
    """Phase 1 verifier stub.

    Returns a trivial "OK" status without performing any checks. Future
    versions will cross-check the answer against retrieved evidence and
    patient context.
    """

    _ = (evidence_bundles, context)  # explicit: unused in Phase 1
    return {"status": "OK", "issues": [], "draft_answer": draft_answer}


# Public orchestrator ------------------------------------------------------


def run_agentic_flow(
    context: QueryContext,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    top_k: int = 5,
    page_range: Optional[Tuple[int, int]] = None,
    llm_client: Optional[OllamaClient] = None,
) -> AgenticResult:
    """Run the Phase 1 agentic flow skeleton.

    Current behaviour:
    - normalizes the input context (no-op)
    - produces a small, deterministic plan (no LLM planner yet)
    - performs per-sub-task retrieval using the provided ``Embedder`` and
      ``VectorStore``
    - returns a clearly marked placeholder answer (reasoning and verification
      remain stubs in this phase)

    This function is safe to call but is not wired into the main analysis
    pipeline yet, so it has no effect on existing behaviour.
    """

    normalized = _normalize_input(context)
    subtasks = _plan_subtasks(normalized)
    evidence = _retrieve_evidence(
        normalized,
        subtasks,
        embedder=embedder,
        vector_store=vector_store,
        top_k=top_k,
        page_range=page_range,
    )
    draft_answer = _reason(normalized, subtasks, evidence, llm_client=llm_client)
    verifier_report = _verify(draft_answer, evidence, normalized)

    debug_info: Dict[str, Any] = {
        "normalized_context": normalized,
        "subtasks": subtasks,
        "evidence_bundles": evidence,
        "verifier_report": verifier_report,
    }

    return AgenticResult(
        final_answer_text=draft_answer,
        used_chunks=[],
        warnings=[
            "agentic flow is in Phase 1 skeleton mode; no real planning/"
            "retrieval/reasoning has been applied",
        ],
        debug_info=debug_info,
    )
