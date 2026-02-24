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

import json
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
    # Rationale: new output requested by updated ARTEMIS instructions. Kept
    # optional and additive to preserve existing behaviour and call-sites.
    updated_management_plan_text: Optional[str] = None
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

    notes_excerpt = patient_notes if patient_notes else ""

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
            notes_text = (ctx.notes_text or "")
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


def _reason_updated_management_plan(
    context: QueryContext,
    evidence_bundles: List[EvidenceBundle],
    *,
    llm_client: Optional[OllamaClient] = None,
) -> Optional[str]:
    # Rationale: Part 3 requires adding a new structured "UPDATED MANAGEMENT PLAN"
    # output alongside the existing toxicity checklist output.
    if llm_client is None:
        return None

    # Rationale: the management plan can be long; request a longer response so the
    # output is less likely to be cut off mid-way.
    _PLAN_CHAT_OPTIONS = {"num_predict": 1600}

    def _strip_code_fences(text: str) -> str:
        # Rationale: tolerate models that wrap JSON in ```json fences.
        t = (text or "").strip()
        if not t.startswith("```"):
            return t
        lines = t.splitlines()
        if not lines:
            return t
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_first_json_object(text: str) -> str:
        # Rationale: models sometimes add extra words before/after the JSON.
        # We extract the first full {...} block so validation can still succeed.
        raw = _strip_code_fences(text)
        if not raw:
            return ""
        start = raw.find("{")
        if start == -1:
            return ""
        depth = 0
        in_str = False
        esc = False
        end = None
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        return raw[start:end] if end is not None else raw[start:]

    def _validate_plan_json(text: str) -> List[str]:
        # Rationale: output must be machine-checkable and reliably structured.
        issues: List[str] = []
        raw = _extract_first_json_object(text)
        if not (text or "").strip():
            return ["empty_output"]
        if not raw:
            # Rationale: caller provided content, but it didn't contain a JSON object.
            return ["no_json_object_found"]

        try:
            obj = json.loads(raw)
        except Exception:
            return ["invalid_json"]

        if not isinstance(obj, dict):
            issues.append("root_not_object")
            return issues

        # Rationale: Part 4 requires an explicit regimen decision (Switch / Hold temporarily / Continue)
        # as a critical decision point.
        required_keys = [
            "art_regimen_decision",
            "problems",
            "monitoring_plan",
            "patient_counselling",
        ]
        for k in required_keys:
            if k not in obj:
                issues.append(f"missing_key:{k}")

        ard = obj.get("art_regimen_decision")
        if not isinstance(ard, dict):
            issues.append("art_regimen_decision_wrong_type")
        else:
            if "decision" not in ard or "reason" not in ard:
                issues.append("art_regimen_decision_missing_fields")
            else:
                decision = str(ard.get("decision") or "").strip()
                allowed = {"Switch", "Hold temporarily", "Continue"}
                if decision not in allowed:
                    issues.append("art_regimen_decision_bad_decision")
                reason = str(ard.get("reason") or "").strip()
                if not reason:
                    issues.append("art_regimen_decision_empty_reason")

        problems = obj.get("problems")
        if not isinstance(problems, list) or not problems:
            issues.append("problems_not_nonempty_list")
        else:
            seen: set[str] = set()
            for idx, p in enumerate(problems[:10]):
                if not isinstance(p, dict):
                    issues.append(f"problem_{idx}_not_object")
                    continue
                for k in ["problem", "action", "reason", "clinician_plan_for_this_problem"]:
                    if k not in p:
                        issues.append(f"problem_{idx}_missing_key:{k}")
                # Heuristic: avoid task-like "problem" naming.
                prob = str(p.get("problem") or "")
                low_prob = prob.strip().lower()
                if (
                    low_prob.startswith("assess")
                    or low_prob.startswith("check")
                    or low_prob.startswith("review")
                    or low_prob.startswith("evaluate")
                    or low_prob.startswith("screen")
                    or low_prob.startswith("monitor")
                ):
                    issues.append(f"problem_{idx}_looks_like_task")

                # Rationale: prevent duplicate problems (common failure mode in draft output).
                norm = " ".join(low_prob.split())
                if norm:
                    if norm in seen:
                        issues.append(f"problem_{idx}_duplicate")
                    seen.add(norm)
                cp = p.get("clinician_plan_for_this_problem")
                if isinstance(cp, dict):
                    if "decision" not in cp or "explanation" not in cp:
                        issues.append(f"problem_{idx}_clinician_plan_missing_fields")
                    else:
                        decision = str(cp.get("decision") or "").strip()
                        allowed = {"Agree", "Disagree", "Gap", "Not Addressed"}
                        if decision not in allowed:
                            issues.append(f"problem_{idx}_clinician_plan_bad_decision")
                elif not isinstance(cp, str):
                    issues.append(f"problem_{idx}_clinician_plan_wrong_type")

        mp = obj.get("monitoring_plan")
        if not isinstance(mp, (str, list, dict)):
            issues.append("monitoring_plan_wrong_type")

        pc = obj.get("patient_counselling")
        if not isinstance(pc, list):
            issues.append("patient_counselling_not_list")

        return issues

    def _organize_plan_json(
        *,
        draft_json_text: str,
        validation_issues: List[str],
    ) -> Optional[str]:
        # Rationale: act as a small "organizer agent" that repairs common
        # structure problems (duplicates, task-like problems, ordering) while
        # keeping the schema unchanged.
        if not draft_json_text.strip():
            return None

        messages = _build_plan_messages(
            retry_feedback=(
                "Draft plan JSON must be reorganized and repaired. Issues: "
                + ", ".join(validation_issues)
                + "\n\n"
                "If the draft is incomplete or invalid, you MUST rebuild a complete, valid plan JSON from scratch. "
                "You MUST de-duplicate repeated problems, merge overlapping ones, "
                "and rewrite problem titles as patient problems/clinical decisions (not tasks). "
                "Order problems from most urgent/high-risk to least. "
                "You MUST include a valid art_regimen_decision with decision one of: Switch, Hold temporarily, Continue. "
                "Keep the plan concise (max 6 problems). "
                "Return ONLY the corrected JSON object."
            )
        )

        # Provide the draft JSON as additional context (user message) so the
        # organizer can repair it deterministically.
        messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    # Rationale: provide only the JSON object (when present) to reduce confusion.
                    "Draft management plan JSON (repair this):\n"
                    + (_extract_first_json_object(draft_json_text) or _strip_code_fences(draft_json_text))
                ),
            }
        ]

        repaired = llm_client.chat(
            messages,
            format="json",
            options_override=_PLAN_CHAT_OPTIONS,
        )
        if repaired is None:
            return None
        repaired = _strip_code_fences(repaired)
        issues = _validate_plan_json(repaired)
        if issues:
            return None
        # Rationale: return only the extracted JSON so downstream renderers don't see extra text.
        return _extract_first_json_object(repaired) or repaired

    def _build_plan_messages(*, retry_feedback: Optional[str] = None) -> List[Dict[str, str]]:
        system_text = (
            "You are an HIV clinician assistant. Using ONLY the provided guideline excerpts and "
            "the patient data in the user message, produce an UPDATED MANAGEMENT PLAN.\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "- Output ONLY a single JSON object. Do NOT output markdown, code fences, or any extra text.\n"
            "- Do NOT include analysis, thinking, rationale preambles, or meta commentary.\n"
            "- Do NOT output any placeholders or template examples.\n"
            "- Problems must be PATIENT PROBLEMS / CLINICAL DECISIONS (not tasks like 'Assess', 'Check', 'Review').\n"
            "- Every Action must be specific enough that a clinician can act on it.\n"
            "- Every Reason must use patient facts and be supported by the provided guideline excerpts.\n"
            "- If you cannot justify an action from the excerpts, write: Uncertain: why you are uncertain (do not guess).\n\n"
            "REQUIRED JSON KEYS (root object):\n"
            "- art_regimen_decision: object with keys decision + reason (decision must be exactly one of: Switch, Hold temporarily, Continue)\n"
            "- problems: array of problem objects (in priority order)\n"
            "- monitoring_plan: string OR array describing what labs to repeat, when, and what result triggers action\n"
            "- patient_counselling: array of strings\n\n"
            "REQUIRED KEYS (art_regimen_decision):\n"
            "- decision: one of Switch, Hold temporarily, Continue\n"
            "- reason: string (patient facts + guideline-grounded OR Uncertain: ...)\n\n"
            "REQUIRED KEYS (each item in problems):\n"
            "- problem: string (a real patient problem / clinical decision)\n"
            "- action: string (specific action OR Uncertain: ...)\n"
            "- reason: string (patient facts + guideline-grounded OR Uncertain: ...)\n"
            "- clinician_plan_for_this_problem: object with keys:\n"
            "  - decision: one of Agree, Disagree, Gap, Not Addressed\n"
            "  - explanation: string\n"
        )

        if retry_feedback:
            system_text += (
                "\n\nFORMAT FIX REQUIRED:\n"
                f"{retry_feedback}\n"
                "Rewrite the plan to fully comply. Output ONLY the corrected plan."
            )

        patient = context.patient_raw or {}
        question = (context.question_text or "").strip() or (
            "Clinical guideline support for this patient."
        )
        patient_id = patient.get("patient_id")
        patient_name = patient.get("name")

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

        regimen_str = ", ".join(patient.get("art_regimen_current") or []) or "unknown"
        notes_text = ""
        try:
            ctx = build_patient_context(patient)
            notes_text = (ctx.notes_text or "")
        except Exception:
            pass

        evidence_lines: List[str] = []
        for bundle in evidence_bundles:
            chunks = list(bundle.chunks or [])
            if not chunks:
                continue
            evidence_lines.append(f"Subtask {bundle.subtask_name}:")
            # Rationale: keep the evidence snippet short to reduce the chance the LLM output
            # is cut off mid-plan.
            for r in chunks[:1]:
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
                evidence_lines.append(f"- page {page}, chunk {chunk_id}: {preview}")
            evidence_lines.append("")
        evidence_text = "\n".join(evidence_lines) or "No evidence retrieved."

        return [
            {"role": "system", "content": system_text},
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

    try:
        messages = _build_plan_messages()
        plan_answer = llm_client.chat(
            messages,
            format="json",
            options_override=_PLAN_CHAT_OPTIONS,
        )
        if plan_answer is None:
            return None

        issues = _validate_plan_json(plan_answer)
        if not issues:
            # Rationale: return only the JSON object (ignore any extra words around it).
            return _extract_first_json_object(plan_answer) or _strip_code_fences(plan_answer)

        # One retry with explicit format feedback (non-blocking if it still fails).
        feedback = "Previous output failed validation: " + ", ".join(issues)
        retry_messages = _build_plan_messages(retry_feedback=feedback)
        plan_retry = llm_client.chat(
            retry_messages,
            format="json",
            options_override=_PLAN_CHAT_OPTIONS,
        )
        if plan_retry is None:
            return plan_answer

        retry_issues = _validate_plan_json(plan_retry)
        if not retry_issues:
            # Rationale: return only the JSON object (ignore any extra words around it).
            return _extract_first_json_object(plan_retry) or _strip_code_fences(plan_retry)

        # Organizer pass: attempt to repair/organize into a valid, de-duplicated
        # management plan JSON.
        organized = _organize_plan_json(
            draft_json_text=plan_retry,
            validation_issues=retry_issues,
        )
        if organized is not None:
            return organized

        # If still invalid, return the latest raw output for debugging/display.
        return _strip_code_fences(plan_retry)
    except Exception:
        return None


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
    updated_management_plan_text = _reason_updated_management_plan(
        normalized,
        evidence,
        llm_client=llm_client,
    )
    verifier_report = _verify(draft_answer, evidence, normalized)

    debug_info: Dict[str, Any] = {
        "normalized_context": normalized,
        "subtasks": subtasks,
        "evidence_bundles": evidence,
        "verifier_report": verifier_report,
        # Rationale: store new Part 3 output for debug UI consumption without
        # changing existing behaviour.
        "updated_management_plan_text": updated_management_plan_text,
    }

    return AgenticResult(
        final_answer_text=draft_answer,
        updated_management_plan_text=updated_management_plan_text,
        used_chunks=[],
        warnings=[
            "agentic flow is in Phase 1 skeleton mode; no real planning/"
            "retrieval/reasoning has been applied",
        ],
        debug_info=debug_info,
    )
