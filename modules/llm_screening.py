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


def _parse_issues_json(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except Exception:
        s = text.strip()
        if s.startswith("```"):
            s = s.split("```", 1)[1]
            if "\n" in s:
                s = s.split("\n", 1)[1]
            if "```" in s:
                s = s.rsplit("```", 1)[0]
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = s[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                return None
        else:
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
        "content": ( """You are a clinical decision support system for HIV medication safety. Your task is to identify discrepancies between a patient's current clinical state and the treatment guidelines provided.

        ## INPUT STRUCTURE
        1. **PATIENT SUMMARY**: Current medications, duration on each drug, lab results, symptoms, examination findings, and relevant history
        2. **GUIDELINE EXCERPTS**: Retrieved sections from clinical guidelines (each tagged with page_number and chunk_id)

        ## YOUR REASONING PROCESS

        ### Step 1: Extract Rules from Guidelines
        First, read the guideline excerpts carefully and extract ALL actionable clinical rules. For each rule, identify:
        - **Trigger condition**: What patient state activates this rule? (e.g., "patient on Drug X", "lab value above/below threshold")
        - **Required action or contraindication**: What should or shouldn't happen?
        - **Monitoring requirements**: What labs/assessments are needed and how often?
        - **Toxicity markers**: What signs/symptoms/labs indicate drug-related harm?

        Look for rules about:
        - Drug contraindications (when a drug should NOT be used)
        - Drug-lab interactions (lab values that make a drug unsafe)
        - Drug-disease interactions (conditions that make a drug unsafe)
        - Drug-drug interactions
        - Required baseline and ongoing monitoring
        - Toxicity thresholds and warning signs
        - When to switch or discontinue therapy

        ### Step 2: Extract Patient State
        Parse the patient summary to identify:
        - Current medications and duration on each
        - All lab values (current and trends if available)
        - Active symptoms and examination findings
        - Relevant comorbidities and history
        - What monitoring has or hasn't been done

        ### Step 3: Systematic Comparison
        For EACH rule extracted from the guidelines:
        1. Check if the rule applies to this patient (e.g., is patient on the relevant drug?)
        2. If applicable, check if the patient's current state violates the rule
        3. Check for COMBINATIONS of factors that together indicate a problem, even if each alone might not

        Pay special attention to:
        - **Compounding risks**: Multiple factors that together create higher risk than each alone
        - **Subclinical findings**: Lab abnormalities even without symptoms
        - **Temporal patterns**: Long duration on a drug increasing toxicity risk
        - **Missing data**: Required monitoring that hasn't been done

        ### Step 4: Generate Alerts
        For each identified discrepancy between patient state and guidelines:
        - Cite the specific guideline rule being violated
        - Explain the clinical reasoning connecting patient findings to the rule
        - Assess severity based on guideline language (words like "discontinue immediately", "contraindicated", "caution" indicate different urgency levels)

        ## SEVERITY DETERMINATION (from guideline language)
        - **critical**: Guidelines use terms like "contraindicated", "discontinue", "do not use", "immediately", or describe life-threatening risks
        - **high**: Guidelines use terms like "avoid", "switch", "significant risk", "closely monitor"
        - **moderate**: Guidelines use terms like "caution", "monitor", "consider", "may cause"
        - **low**: Guidelines suggest optimization or best practice without safety concern

        ## OUTPUT FORMAT
        
        Return ONLY a valid JSON array.
 
        - If a guideline-supported issue exists in the provided excerpts: output it with citations.
        - If patient facts suggest a potentially important issue but the provided excerpts do NOT contain a supporting rule:
        output an item with issue_type="information_gap", citations=[], and clearly state
        "Not supported by provided excerpts; needs more guideline retrieval."
        - Output [] only if:
        (a) no guideline-supported issues are found AND
        (b) no information gaps / potential concerns requiring more guideline evidence are present.
        Never invent citations.

        ```json
        [
        {
            "alert_id": "unique_string",
            "title": "Brief description of the issue",
            "message": "Detailed explanation: (1) what guideline rule applies, (2) what patient findings violate it, (3) why this is clinically significant",
            "issue_type": "medication_safety|monitoring_gap|toxicity|drug_interaction|guideline_deviation",
            "severity": "critical|high|moderate|low",
            "recommended_action": "Specific action based on guideline recommendation",
            "citations": [{"page_number": int, "chunk_id": "string"}]
        }
        ]
        ```

        ## CRITICAL INSTRUCTIONS

        1. **Do NOT rely on your training knowledge for clinical rules** - use ONLY the provided guideline excerpts to determine what constitutes a problem
        2. **Be explicit about your reasoning** - in the message field, quote or paraphrase the specific guideline language that informs your alert
        3. **Connect the dots** - if guidelines say "Drug X causes Condition Y" and patient has both Drug X and signs of Condition Y, flag it even if not explicitly stated as a contraindication
        4. **Check combinations** - a drug + abnormal lab + contributing comorbidity may together constitute a problem per guidelines
        5. **Flag missing monitoring** - if guidelines specify required monitoring and it's absent from patient data, that's a gap
        6. **When uncertain, flag with lower severity** rather than missing a potential issue

        ## EXAMPLE REASONING (do not use these specific rules - extract rules from provided guidelines)

        If guidelines state: "Monitor hemoglobin at baseline and regularly for patients on [Drug]. Discontinue if hemoglobin falls below [threshold] or in patients with conditions causing blood loss."

        And patient has: [Drug] for 4 years, hemoglobin below threshold, AND a condition causing ongoing blood loss

        Then: Flag as critical because (1) patient is on the drug, (2) lab is below discontinuation threshold, (3) compounding factor (blood loss) is present, making the situation more urgent per guideline logic.

        Remember: Your job is to be a careful reader of guidelines and a systematic checker of patient data against those guidelines. The guidelines are your source of truth."""
        ),
    }
    

    user: ChatMessage = {
        "role": "user",
        "content": (
            f"Patient: {patient_context.name} ({patient_context.patient_id})\n"
            f"Encounter date: {patient_context.encounter_date.isoformat()}\n\n"
            "Raw clinical history (concatenated notes_text):\n"
            f"{patient_context.notes_text}\n\n"
            "Stage 1 summary JSON:\n"
            f"{json.dumps(stage1_summary, ensure_ascii=False)}\n\n"
            "Guideline evidence excerpts (screening bundle):\n"
            f"{json.dumps(bundle, ensure_ascii=False)}\n\n"
            "Now output ONLY a JSON array of issues as described, or an empty array if there are no clinically relevant issues."
        ),
    }

    # Rationale: increase num_predict so the model is less likely to truncate a long JSON array.
    # Keep a safe fallback for clients/tests that don't accept the structured-output kwargs.
    try:
        llm_text = llm_client.chat(
            [system, user],
            format=output_schema,
            options_override={"num_predict": 768},
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
