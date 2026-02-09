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
from modules.patient_parser import build_patient_context, compute_lab_trends
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
    """Patient-aware planner for toxicity-focused light decomposition.

    Rationale: instead of always asking the same generic question, the planner
    inspects the patient's current drugs and creates a targeted retrieval
    sub-task *per drug* so the vector store returns the most relevant guideline
    sections about each drug's known side effects and toxicity markers.  It also
    adds a symptom-to-toxicity matching task and a doctor's-plan gap check.
    """

    question = (context.question_text or "").strip()
    if not question:
        question = "Clinical guideline support for this patient."

    subtasks: List[SubTaskPlan] = []

    # --- Suggestion 3: per-drug toxicity retrieval ---
    # Rationale: pulling guideline sections per drug maximizes the chance of
    # surfacing drug-specific side effects, monitoring requirements, and
    # switching criteria that a generic query would miss.
    regimen: List[str] = []
    try:
        regimen = list(context.patient_raw.get("art_regimen_current") or [])
    except Exception:
        pass

    for drug in regimen:
        subtasks.append(
            SubTaskPlan(
                name=f"drug_toxicity_{drug}",
                description=(
                    f"Retrieve guideline sections about known side effects, "
                    f"toxicity markers, monitoring requirements, and switching "
                    f"criteria for {drug}."
                ),
                retrieval_query=(
                    f"{drug} side effects toxicity monitoring renal hepatic "
                    f"haematological switching criteria contraindications"
                ),
                priority="high",
            )
        )

    # --- Symptom-to-toxicity matching task ---
    # Rationale: the doctor may have documented symptoms (e.g. fatigue, bone
    # pain, reduced urine output) that could be early signs of drug toxicity.
    # This sub-task retrieves guideline content about symptom-based toxicity
    # recognition so the LLM can cross-check patient symptoms against known
    # drug side effects.
    subtasks.append(
        SubTaskPlan(
            name="symptom_toxicity_match",
            description=(
                "Retrieve guideline sections that describe how specific "
                "symptoms (e.g. fatigue, bone pain, rash, reduced urine, "
                "nausea, peripheral neuropathy) can indicate ART drug "
                "toxicity, and how to confirm each suspected toxicity."
            ),
            retrieval_query=(
                f"{question}\n\n"
                "Focus on: which symptoms indicate drug toxicity, how to "
                "confirm suspected toxicity with lab tests, and what "
                "confirmatory investigations are recommended."
            ),
            priority="high",
        )
    )

    # --- Suggestion 4: doctor's plan gap check ---
    # Rationale: the user's workflow explicitly states that MedGemma should run
    # *after* the doctor has made a plan, and flag anything the plan missed.
    # This sub-task retrieves guideline actions so the LLM can compare them
    # against the doctor's documented plan.
    subtasks.append(
        SubTaskPlan(
            name="doctors_plan_check",
            description=(
                "Retrieve guideline sections about recommended clinical "
                "actions, required monitoring, counselling, and follow-up "
                "so the system can compare them against the doctor's "
                "current plan and flag any gaps."
            ),
            retrieval_query=(
                f"{question}\n\n"
                "Focus on: recommended actions after identifying drug "
                "toxicity, required lab monitoring intervals, patient "
                "counselling, when to switch regimen, and follow-up timing."
            ),
            priority="medium",
        )
    )

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
            for r in chunks[:3]:
                try:
                    metadata = getattr(r, "metadata", {}) or {}
                    page = metadata.get("page_number")
                    chunk_id = getattr(r, "chunk_id", None)
                    doc_text = getattr(r, "document", "") or ""
                except Exception:
                    continue
                preview = doc_text.replace("\n", " ")
                if len(preview) > 300:
                    preview = preview[:300] + "…"
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

        # --- Suggestion 2 (wired): compute lab trends and include them ---
        # Rationale: doctors spot toxicity by seeing trends over time (e.g.
        # rising creatinine).  We compute the trends here and inject them
        # into the user message so the LLM has this information explicitly.
        lab_trends_text = ""
        try:
            raw_labs = patient.get("labs") or {}
            encounter_date_str = (patient.get("today_encounter") or {}).get("date")
            enc_date = None
            if encounter_date_str:
                from datetime import datetime as _dt
                enc_date = _dt.strptime(encounter_date_str, "%Y-%m-%d").date()
            trends = compute_lab_trends(raw_labs, encounter_date=enc_date)
            if trends:
                lab_trends_text = "\n".join(t.summary_text for t in trends)
        except Exception:
            lab_trends_text = ""

        # --- Suggestion 4 (wired): extract doctor's current plan ---
        # Rationale: the user's workflow says MedGemma should check *after*
        # the doctor has made a plan.  We surface that plan so the LLM can
        # evaluate whether it addresses all identified issues.
        doctors_plan = ""
        try:
            today = patient.get("today_encounter") or {}
            plan_parts: List[str] = []
            note = (today.get("note") or "").strip()
            if note:
                plan_parts.append(f"Today's note: {note}")
            orders = today.get("orders") or []
            if orders:
                plan_parts.append(f"Orders: {orders}")
            med_changes = today.get("med_changes") or []
            if med_changes:
                plan_parts.append(f"Medication changes: {med_changes}")
            doctors_plan = "\n".join(plan_parts) if plan_parts else "No plan documented."
        except Exception:
            doctors_plan = "No plan documented."

        # Regimen and notes for context in the user message.
        regimen_str = ", ".join(patient.get("art_regimen_current") or []) or "unknown"
        notes_text = ""
        try:
            ctx = build_patient_context(patient)
            notes_text = (ctx.notes_text or "").strip()[:600]
        except Exception:
            pass

        # --- Suggestion 5: structured toxicity-focused system prompt ---
        # Rationale: instead of free-text output, we ask the LLM to produce a
        # structured checklist that maps each suspected toxicity to supporting
        # evidence, confirmation steps, recommended actions, and gaps in the
        # doctor's plan.  This makes the output actionable and easy to review.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a drug toxicity screening assistant for an HIV "
                    "clinician.  Your job is to catch toxicities the doctor "
                    "may have missed.  Think step-by-step:\n\n"
                    "STEP 1 — SYMPTOM SCAN\n"
                    "List every symptom, exam finding, and lab abnormality "
                    "from the patient data.  For labs, pay close attention to "
                    "TRENDS over time (values going up or down across visits).\n\n"
                    "STEP 2 — DRUG-TOXICITY MATCHING\n"
                    "For each finding from Step 1, check whether ANY of the "
                    "patient's current (or recent past) medications could cause "
                    "it, using ONLY the provided guideline excerpts.  Also "
                    "check for contributing co-medications (e.g. NSAIDs, TB "
                    "drugs) mentioned in the notes.\n\n"
                    "STEP 3 — CONFIRMATION\n"
                    "For each suspected toxicity, state what test or "
                    "investigation the guidelines say is needed to confirm it.\n\n"
                    "STEP 4 — DOCTOR'S PLAN GAP CHECK\n"
                    "Compare the doctor's current plan against what the "
                    "guidelines recommend.  Flag anything the doctor's plan "
                    "is missing (e.g. a regimen switch, stopping a nephrotoxic "
                    "drug, ordering a specific lab, counselling, follow-up "
                    "timing).\n\n"
                    "OUTPUT FORMAT — For each suspected issue, output:\n"
                    "- Suspected toxicity: [name]\n"
                    "- Supporting evidence: [patient findings + lab trends]\n"
                    "- Guideline basis: [what the guideline excerpt says]\n"
                    "- Confirmation needed: [test/investigation]\n"
                    "- Recommended action: [what guidelines say to do]\n"
                    "- Doctor's plan gap: [what is missing from current plan]\n\n"
                    "If no toxicity is suspected, say so clearly.\n"
                    "If the guideline excerpts are insufficient, say you are "
                    "unsure instead of guessing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient id: {patient_id}, name: {patient_name}\n"
                    f"Current regimen: {regimen_str}\n"
                    f"Question: {question}\n\n"
                    f"Patient notes:\n{notes_text}\n\n"
                    f"Lab trends:\n{lab_trends_text or 'No lab history available.'}\n\n"
                    f"Doctor's current plan:\n{doctors_plan}\n\n"
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
