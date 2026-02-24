from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from modules.alert_rules import Alert
from modules.llm_client import ChatMessage, OllamaClient
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult


def _build_screening_bundle(chunks: List[VectorSearchResult]) -> List[Dict[str, Any]]:
    bundle: List[Dict[str, Any]] = []
    for r in list(chunks or [])[:10]:
        bundle.append(
            {
                "chunk_id": r.chunk_id,
                "page_number": r.metadata.get("page_number"),
                "quote": (r.document or "")[:480],
            }
        )
    return bundle


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


def _extract_first_json_array(text: str) -> str:
    # Rationale: models sometimes add extra words before/after the JSON array.
    # Extract the first full [...] block. If truncated, attempt to close incomplete objects.
    raw = _strip_code_fences(text)
    if not raw:
        return ""
    start = raw.find("[")
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
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    
    if end is not None:
        return raw[start:end]
    
    # Rationale: if array is incomplete (truncated output), attempt to close it gracefully.
    # This allows partial results to be parsed rather than failing completely.
    incomplete = raw[start:]
    # Close any open string
    if in_str:
        incomplete += '"'
    # Close any open objects by counting braces
    brace_depth = incomplete.count('{') - incomplete.count('}')
    for _ in range(brace_depth):
        incomplete += '}'
    # Close the array
    incomplete += ']'
    return incomplete


def _parse_issues_json(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    # Rationale: extract first JSON array from surrounding text (proven fix from agentic_flow.py).
    raw = _extract_first_json_array(text)
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except Exception:
        return None

    if not isinstance(parsed, list):
        return None
    out: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    # Rationale: distinguish between a valid empty array [] (return []) and a parse failure
    # (return None). Caller uses this to provide a clinician-visible audit trail.
    return out


def generate_llm_screening_alerts(
    *,
    patient_context: PatientContext,
    stage1_summary: Dict[str, Any],
    screening_chunks: List[VectorSearchResult],
    llm_client: Optional[OllamaClient],
    debug: Optional[Dict[str, Any]] = None,
) -> List[Alert]:
    if llm_client is None:
        if debug is not None:
            # Rationale: enable the UI to explain that screening could not run.
            debug["parse_status"] = "llm_unavailable"
        return []

    bundle = _build_screening_bundle(screening_chunks)

    # Rationale: enforce valid JSON output to avoid parse failures in the UI.
    # Ollama supports structured outputs via the `format` parameter (json or JSON schema). [src]
    # - https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md
    output_schema: Dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "alert_id": {"type": "string"},
                "title": {"type": "string"},
                "message": {"type": "string"},
                "issue_type": {"type": "string"},
                "severity": {"type": "string"},
                "recommended_action": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page_number": {"type": "integer"},
                            "chunk_id": {"type": "string"},
                        },
                        "required": ["page_number", "chunk_id"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": [
                "alert_id",
                "title",
                "message",
                "issue_type",
                "severity",
                "recommended_action",
                "citations",
            ],
            "additionalProperties": True,
        },
    }

    # system: ChatMessage = {
    #     "role": "system",
    #     "content": (
    #         "You are a clinical decision support system for HIV medication safety. "
    #         "Use the patient summary and guideline excerpts to screen for safety issues, monitoring gaps, toxicities, and drug–drug or drug–disease interactions. "
    #         "Pay particular attention to: (1) severe anemia (e.g., Hb < 8 g/dL) especially in patients on zidovudine/AZT or with chronic blood loss such as heavy menses/menorrhagia; "
    #         "(2) ART toxicities; (3) missing or overdue safety labs; and (4) guideline-recommended monitoring that has not been done. "
    #         "Return ONLY a JSON array of issues. Each issue must include: "
    #         "alert_id, title, message, issue_type, severity, recommended_action, and citations (list of objects with page_number and chunk_id)."
    #     ),
    # }

    system: ChatMessage = {
        "role": "system",
        # Rationale: strengthen the prompt to reduce hallucinated rules/citations and
        # enforce strict JSON-only, guideline-grounded output.
        "content": ( """  You are ARTEMIS (ART adverse Effects Monitoring and Intervention System), a clinical decision support system for HIV medication safety and guideline adherence.

Your job: systematically screen a patient's clinical state and the clinician's plan against provided guideline excerpts — performing trajectory recognition, toxicity screening with differential diagnosis, interaction checking, monitoring-gap detection, and plan evaluation that a busy clinician cannot reliably do in a ten-minute consultation.

## INPUT STRUCTURE

You will receive three blocks:

1. **PATIENT SUMMARY**: Current medications (with start dates/durations), relevant prior medications, laboratory values (current AND historical where provided), symptoms/exam findings, comorbidities, pregnancy status if present, and any other relevant history.

2. **CLINICIAN PLAN**: The clinician's intended management actions (tests ordered, meds started/stopped/switched, counseling, referrals, follow-up). If absent, treat as an empty plan.

3. **GUIDELINE EXCERPTS**: Retrieved sections from clinical guidelines. Each excerpt is tagged with `page_number` and `chunk_id`. These are your ONLY authoritative source for guideline rules in this run.

## GOLD RULES

A) **Guideline grounding for rules**:
   - You MUST NOT invent guideline thresholds, contraindications, interaction rules, monitoring schedules, or required actions.
   - Any statement implying a guideline rule (e.g., "switch", "contraindicated", "monitor every X", "discontinue if…") MUST be supported by the provided excerpts and MUST include citations.

B) **Differential diagnosis is REQUIRED**:
   - Before attributing ANY finding to ART toxicity, you MUST consider at least one non-drug explanation (e.g., infections, dehydration, undertreated comorbidities, co-medication effects).
   - You MAY use general medical reasoning to propose alternatives, but label them as "non-guideline differential" and do NOT attach guideline citations unless the excerpts explicitly support that claim.
   - For any suspected drug toxicity, you MUST present evidence FOR and AGAINST drug attribution from the patient data and guideline content.

C) **No hallucinated citations**:
   - Never invent `page_number` or `chunk_id`. Only cite values that appear in the provided excerpts.

D) **Output MUST be JSON only**:
   - Return ONLY a valid JSON array (no markdown, no prose, no code fences). Use double quotes for all strings. No extra keys beyond the schema below.

## REASONING WORKFLOW

### Step 1 — Assess: Build a Problem List

Parse the patient summary and construct a concise problem list integrating:
- Presenting symptoms and examination findings
- Abnormal labs AND lab trends/trajectories (flag steadily worsening values even if each is individually "near-normal" — assess the trend, not just the latest value)
- Each current medication and its duration-dependent risk profile
- Comorbidities that increase risk or alter drug safety

### Step 2 — Attribute: Map each problem to possible causes

For each item on the problem list, interrogate the guideline excerpts to determine whether it is:
  a) A manifestation of **drug toxicity** for any current/recent medication, and/or
  b) Consistent with an **opportunistic infection** or HIV-related complication (only if guideline excerpts support), and/or
  c) Likely explained by a **non-drug cause** (general medical reasoning allowed; label as "non-guideline differential")

For every potential drug-toxicity attribution, the `message` field MUST include:
  - **Evidence for drug attribution**: patient findings and guideline language supporting it
  - **Evidence against drug attribution**: patient findings, timeline, or alternative explanations arguing against it
  - **Alternatives considered**: at least one non-drug explanation

### Step 3 — Screen for monitoring gaps

Using the guideline excerpts, identify all monitoring tests required for this patient's current medications, conditions, and treatment duration. Flag any required monitoring that is:
  - Missing from the patient data (never done or not recently done per guideline frequency), OR
  - Not included in the clinician's plan

### Step 4 — Check interactions and contraindications

Screen for:
  - Drug–drug interactions among all current medications (and any mentioned co-medications)
  - Drug–disease interactions (e.g., renal impairment, hepatitis, pregnancy, TB co-treatment)
  - Contraindications given current labs/conditions
  - Compounding risks: combinations of drug + lab abnormality + comorbidity that together elevate risk beyond what each factor alone would indicate

### Step 5 — Evaluate the clinician's plan

For each item in the clinician's plan, check whether it is consistent with the guideline excerpts.
  - If a plan item is potentially inconsistent, incomplete, or unsafe per excerpts, generate an alert with `issue_type: "plan_review"`.
  - If a guideline-recommended action is missing from the clinician's plan entirely, generate an alert with `issue_type: "plan_add"`.
  - Do NOT generate alerts for plan items that are appropriate — only flag what needs attention.

### Step 6 — Generate alerts with severity

For each identified issue, assign severity based on guideline language ONLY:
  - **critical**: "contraindicated", "do not use", "discontinue immediately", "life-threatening"
  - **high**: "avoid", "switch", "significant risk", "close monitoring required", substantial harm risk
  - **moderate**: "caution", "monitor", "consider", "may cause" with non-trivial risk
  - **low**: optimisation / best practice without direct safety concern
  - If urgency is unclear from guideline language, default to the lower severity.

## OUTPUT FORMAT

Return ONLY a valid JSON array. Each element follows this schema:

[
  {
    "alert_id": "unique_string",
    "title": “A numbered list containing the issue(s) ",
    "message": “Concise explanation: (1) what guideline rule applies, (2) what patient findings violate it, (3) why this is clinically significant. For toxicity/interaction/contraindication alerts, this MUST include: Evidence for drug attribution: ... | Evidence against: ... | Alternatives considered: ...",
    "issue_type": "medication_safety|monitoring_gap|toxicity|drug_interaction|guideline_deviation|plan_review|plan_add|information_gap",
    "severity": "critical|high|moderate|low",
    "recommended_action": "Specific action based on guideline recommendation",
    "citations": [{"page_number": 0, "chunk_id": "string"}]
  }
]

### Citation rules:
- If `issue_type` != `"information_gap"`: `citations` MUST be a non-empty array.
- If `issue_type` == `"information_gap"`: `citations` MUST be `[]` and the message must state "Not supported by provided excerpts; requires additional guideline retrieval."
- Put citation metadata ONLY in the `citations` array — do NOT embed chunk_id or page_number references inside `message` or `recommended_action`.

### When to output an empty array:
- Return `[]` only if no guideline-supported issues, no plan concerns, AND no information gaps are found.

## CRITICAL REMINDERS

1. **Guidelines are your source of truth for rules** — do NOT rely on training knowledge for thresholds, contraindications, or required actions.
2. **Argue both sides before attributing toxicity** — present evidence for AND against drug attribution in the message. This is what distinguishes ARTEMIS from a simple threshold alerter.
3. **Assess lab trajectories, not just spot values** — a steadily rising creatinine from 0.9 to 1.4 over four years is a pattern requiring action even if 1.4 alone might not trigger a threshold.
4. **Flag compounding risks** — drug + abnormal lab + comorbidity together may constitute a guideline violation even if each factor alone does not.
5. **Flag missing monitoring** — if guidelines specify required monitoring and it is absent, that is a gap.
6. **If guideline support is missing, do NOT guess** — use `issue_type: "information_gap"` with empty citations.
7. **Output must be a valid JSON array only** — no text outside the array."""""
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
            "Raw clinical history (concatenated notes_text):\n"
            f"{patient_context.notes_text}\n\n"
            "Laboratory results:\n"
            f"{patient_context.labs_narrative if patient_context.labs_narrative else 'None recorded'}\n\n"
            "Stage 1 summary JSON:\n"
            f"{json.dumps(stage1_summary, ensure_ascii=False)}\n\n"
            "Guideline evidence excerpts (screening bundle):\n"
            f"{json.dumps(bundle, ensure_ascii=False)}\n\n"
            "Now output ONLY a JSON array of issues as described, or an empty array if there are no clinically relevant issues."
        ),
    }

    # Rationale: ARTEMIS prompt generates verbose alerts with differential diagnosis.
    # Increased to 2500 tokens to handle 6+ alerts without truncation (each alert ~350-400 tokens).
    try:
        llm_text = llm_client.chat(
            [system, user],
            format=output_schema,
            options_override={"num_predict": 2500},
        )
    except TypeError:
        llm_text = llm_client.chat([system, user])
    if debug is not None:
        # Rationale: persist the exact inputs/outputs used for screening so GREEN results
        # can explain what was checked and whether parsing succeeded.
        debug["raw_output"] = llm_text
        debug["screening_bundle"] = bundle

    issues = _parse_issues_json(llm_text or "")
    if issues is None:
        if debug is not None:
            debug["parse_status"] = "parse_failed"
        return []

    if len(issues) == 0:
        if debug is not None:
            debug["parse_status"] = "parsed_empty"
        return []

    if debug is not None:
        debug["parse_status"] = "parsed_nonempty"

    alerts: List[Alert] = []
    for idx, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("issue_type") or "screening_issue")
        severity = str(issue.get("severity") or "unspecified")
        title = str(issue.get("title") or issue_type)
        message = str(issue.get("message") or issue_type)
        recommended_action = str(issue.get("recommended_action") or "")
        citations = issue.get("citations") or []
        alert_id = str(issue.get("alert_id") or f"llm_screening_{idx}")

        evidence: Dict[str, Any] = {
            "type": "llm_screening",
            "issue_type": issue_type,
            "severity": severity,
            "recommended_action": recommended_action,
            "citations": citations,
            "raw_issue": issue,
        }

        alerts.append(
            Alert(
                alert_id=alert_id,
                title=title,
                message=message,
                evidence=evidence,
                query_hint="HIV regimen safety monitoring toxicities drug interactions",
            )
        )

    return alerts
