from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging
import re
from difflib import SequenceMatcher

from modules.alert_rules import Alert
from modules.llm_client import ChatMessage, OllamaClient
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult

logger = logging.getLogger(__name__)


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
    
    # Rationale: if array is incomplete (truncated output), salvage all fully-formed
    # objects and discard the last incomplete one. Simply closing braces on a truncated
    # object (e.g., '{"page') produces invalid JSON like '{"page}]'.
    # Instead, find the last complete '}' that ends a top-level object and cut there.
    incomplete = raw[start:]
    # Find the position of the last '}' that closes a complete top-level object
    last_complete_obj = incomplete.rfind('},')
    if last_complete_obj == -1:
        last_complete_obj = incomplete.rfind('}')
    if last_complete_obj != -1:
        # Keep everything up to and including the last complete '}'
        incomplete = incomplete[:last_complete_obj + 1] + ']'
    else:
        # No complete objects at all — return empty array
        incomplete = '[]'
    # Rationale: fix trailing commas before ] that json.loads rejects
    incomplete = re.sub(r',\s*\]', ']', incomplete)
    return incomplete


def _deduplicate_message(text: str, max_words: int = 200) -> str:
    """Inline dedup + word limit — guaranteed to run on every alert message.
    
    Rationale: the artemis_postprocess pipeline may fail silently (import error,
    JSON parse error, etc.) and the fallback parser returns raw LLM text with no
    dedup. This function is the last line of defense against repetition loops.
    
    Two-layer approach:
      Layer 1 — Phrase-level: detect any phrase (5+ words) that repeats 3+ times
                and collapse it to a single occurrence. This catches loops like
                "The patient is on TDF, 3TC, and DTG. The patient is on TDF, 3TC..."
      Layer 2 — Sentence-level: split on sentence boundaries and remove sentences
                that are ≥75% similar to one already kept.
    """
    if not text:
        return text
    
    orig_len = len(text.split())
    
    # --- Layer 1: Phrase-level repetition collapse ---
    # Rationale: the LLM sometimes gets stuck repeating a phrase that doesn't end
    # with punctuation, so sentence splitting can't catch it. We look for any
    # substring of 5+ words that appears 3+ times and keep only the first occurrence.
    # This regex finds a phrase of 5+ words that repeats at least 2 more times.
    text = re.sub(
        r'((?:\S+\s+){4,}\S+[.,;:!?]?\s*?)(\1{2,})',
        r'\1',
        text,
    )
    
    # --- Layer 2: Sentence-level deduplication ---
    # Split into sentences at period/exclamation/question followed by space + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    
    kept: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Check if this sentence is too similar to one we already kept
        is_dup = False
        for existing in kept:
            # Rationale: SequenceMatcher ratio >= 0.75 catches near-identical sentences
            # that differ only in minor wording (e.g., rounding differences).
            if SequenceMatcher(None, sentence.lower(), existing.lower()).ratio() >= 0.75:
                is_dup = True
                break
        if not is_dup:
            kept.append(sentence)
    
    result = " ".join(kept)
    
    # Enforce word limit with sentence-boundary truncation
    words = result.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        last_period = truncated.rfind(".")
        if last_period > len(truncated) * 0.5:
            truncated = truncated[:last_period + 1]
        result = truncated
    
    new_len = len(result.split())
    if new_len < orig_len:
        logger.info(
            "Dedup reduced message: %d→%d words (%d chars→%d chars)",
            orig_len, new_len, len(text), len(result),
        )
    return result


def _parse_issues_json(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    # Rationale: use artemis_postprocess pipeline for comprehensive deduplication and validation.
    # This catches repetition loops that prompt engineering misses (safety net).
    try:
        from modules.artemis_postprocess import clean_artemis_output
        
        # clean_artemis_output handles: JSON extraction, sentence/paragraph/alert dedup,
        # word limit enforcement, schema validation
        cleaned_alerts = clean_artemis_output(
            text,
            message_max_words=200,
            action_max_words=50,
            sentence_similarity=0.75,
            paragraph_similarity=0.70,
            alert_similarity=0.70,
        )
        
        # Return None only if parsing completely failed (clean_artemis_output returns error alert)
        if len(cleaned_alerts) == 1 and cleaned_alerts[0].get("alert_id") == "PARSE_ERROR":
            return None
        
        return cleaned_alerts if cleaned_alerts else []
        
    except Exception as exc:
        # Rationale: catch ALL exceptions so that any failure in the postprocessing pipeline
        # falls back to the proven original parser instead of crashing.
        # Log the error so we can diagnose why postprocessing failed.
        logger.warning("artemis_postprocess failed, using fallback parser: %s", exc)

    # Fallback: original robust parsing (no dedup, but at least parses the JSON)
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

    # Rationale: ARTEMIS v3 prompt — redesigned for 4B quantized models.
    # Key design principles for small models:
    #   1. SHORT (~600 words vs ~1500 in v2) — small models lose track after ~500 words
    #   2. FLAT structure — no deep nesting, numbered lists only
    #   3. EXAMPLE-ANCHORED — one concrete example alert to guide format
    #   4. END-WEIGHTED — most critical rules at the end (small models attend to end)
    #   5. NO internal reasoning chains — just describe the desired output
    # All clinical screening requirements preserved: toxicity, interactions,
    # monitoring gaps, differential diagnosis, severity, citations.
    system: ChatMessage = {
        "role": "system",
        "content": """You are ARTEMIS, an HIV medication safety screening system. You receive a patient summary, guideline excerpts (with page_number and chunk_id), and optionally a clinician plan. You output ONLY a JSON array of safety alerts.

WHAT TO SCREEN FOR:
1. Drug toxicities — check if any lab abnormality or symptom could be caused by a current medication. Consider non-drug causes too (infection, dehydration, comorbidity). Present evidence for AND against drug attribution.
2. Drug interactions — between current medications, and between drugs and conditions (renal impairment, pregnancy, TB co-treatment).
3. Monitoring gaps — required labs or tests that are missing, overdue, or not in the clinician's plan.
4. Lab trends — a steadily worsening value (e.g., creatinine rising from 0.9 to 1.4 over years) is a problem even if the latest value alone seems acceptable.
5. Plan problems — anything in the clinician's plan that conflicts with guidelines, or guideline actions missing from the plan.
6. Compounding risks — drug + abnormal lab + comorbidity together may be dangerous even if each factor alone is not.

RULES:
- Only cite page_number and chunk_id values that appear in the provided excerpts. Never invent citations.
- Only state guideline rules that the excerpts support. Do not invent thresholds or contraindications.
- If guideline support is missing, use issue_type "information_gap" with empty citations.
- Merge related excerpts: if 3 excerpts discuss the same topic, write ONE synthesized statement and list all citations.

SEVERITY:
- critical: "contraindicated", "do not use", "discontinue immediately"
- high: "avoid", "switch", "significant risk", substantial harm
- moderate: "caution", "monitor", "consider"
- low: optimisation / best practice

OUTPUT FORMAT — return ONLY a JSON array:
[
  {
    "alert_id": "unique_string",
    "title": "Short title, max 15 words",
    "message": "Max 150 words. State the guideline rule, then patient findings, then for toxicity alerts: evidence for and against drug attribution. Each sentence must add new information.",
    "issue_type": "medication_safety|monitoring_gap|toxicity|drug_interaction|guideline_deviation|plan_review|plan_add|information_gap",
    "severity": "critical|high|moderate|low",
    "recommended_action": "Specific guideline-based action, max 40 words",
    "citations": [{"page_number": 0, "chunk_id": "string"}]
  }
]

EXAMPLE ALERT (use real data and real chunk_ids from the excerpts, not these placeholder values):
{"alert_id": "EXAMPLE_001", "title": "Example Drug Side Effect", "message": "Guidelines state X. Patient shows Y. Drug A is a known cause. However, non-drug cause B should also be considered.", "issue_type": "toxicity", "severity": "high", "recommended_action": "Specific action from the guideline excerpts.", "citations": [{"page_number": 99, "chunk_id": "use_real_chunk_id_from_excerpts"}]}

HARD LIMITS — FOLLOW STRICTLY:
- Maximum 5 alerts total. If you find more than 5 issues, keep only the most clinically important ones.
- If two issues are closely related (e.g., weight gain and obesity, or two aspects of the same drug toxicity), merge them into ONE alert. Do NOT create separate alerts for the same clinical concept.
- Each message: maximum 150 words. Each recommended_action: maximum 40 words.
- NEVER repeat the same point within an alert. Every sentence must add NEW information.
- If you have said something once, do NOT say it again. Stop writing that alert and move to the next one.
- Each alert_id must be unique. Never use the same alert_id twice.
- Return [] if no issues found.
""",
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

    # Rationale: Ollama parameters tuned for 4B quantized model (MedGemma Q4_K_S).
    # - num_predict=1600: enough for ~5 detailed alerts without truncation.
    #   1200 caused truncation on complex patients with 5+ alerts (parse_failed).
    # - repeat_penalty=1.5: strong penalty on recently used tokens to break loops.
    #   [src] https://docs.ollama.com/modelfile — values >1.0 penalize repeated tokens.
    # - repeat_last_n=256: look-back window for repeat penalty. Default is 64 tokens
    #   (~50 words), which is too short to catch phrase-level loops. 256 tokens (~200
    #   words) covers the full length of a typical alert message.
    #   [src] https://docs.ollama.com/modelfile — repeat_last_n controls the window.
    try:
        llm_text = llm_client.chat(
            [system, user],
            format=output_schema,
            options_override={
                "num_predict": 1600,
                "repeat_penalty": 1.5,
                "repeat_last_n": 256,
            },
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
    dedup_stats: List[Dict[str, Any]] = []  # Debug: track dedup effectiveness per alert
    seen_alert_ids: set = set()   # Rationale: catch exact duplicate alert_ids
    seen_titles: List[str] = []   # Rationale: catch near-duplicate alerts by title similarity
    for idx, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("issue_type") or "screening_issue")
        severity = str(issue.get("severity") or "unspecified")
        title = str(issue.get("title") or issue_type)
        message_raw = str(issue.get("message") or issue_type)
        alert_id = str(issue.get("alert_id") or f"llm_screening_{idx}")

        # --- Alert-level dedup: skip duplicate alerts ---
        # Rationale: the model sometimes generates identical alerts (same alert_id)
        # or near-identical alerts with slightly different IDs but same content.
        if alert_id in seen_alert_ids:
            logger.info("Skipping duplicate alert_id: %s", alert_id)
            continue
        is_dup_title = False
        for existing_title in seen_titles:
            if SequenceMatcher(None, title.lower(), existing_title.lower()).ratio() >= 0.75:
                is_dup_title = True
                break
        if is_dup_title:
            logger.info("Skipping near-duplicate alert title: %s", title)
            continue
        seen_alert_ids.add(alert_id)
        seen_titles.append(title)
        recommended_action = str(issue.get("recommended_action") or "")
        citations = issue.get("citations") or []
        
        # Rationale: guaranteed dedup — runs regardless of whether artemis_postprocess succeeded.
        # This is the last line of defense against LLM repetition loops.
        message = _deduplicate_message(message_raw)
        recommended_action = _deduplicate_message(recommended_action, max_words=50)
        
        # Debug: track how much dedup reduced the message
        dedup_stats.append({
            "alert_id": alert_id,
            "raw_words": len(message_raw.split()),
            "clean_words": len(message.split()),
            "raw_chars": len(message_raw),
            "clean_chars": len(message),
        })

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

    # Debug: persist dedup stats so they show in the Streamlit debug expander
    if debug is not None:
        debug["dedup_stats"] = dedup_stats

    return alerts
