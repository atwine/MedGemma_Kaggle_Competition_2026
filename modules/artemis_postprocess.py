"""
ARTEMIS Post-Processing: Repetition Detection & Cleanup
========================================================
Catches and fixes LLM repetition loops in ARTEMIS alert output.
Works as a safety net even when prompt-level controls succeed 95% of the time.

Usage:
    from modules.artemis_postprocess import clean_artemis_output

    raw_response = llm_call(...)   # string from LLM
    alerts = clean_artemis_output(raw_response)
"""

import json
import re
import logging
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any

logger = logging.getLogger("artemis_postprocess")


# ---------------------------------------------------------------------------
# 1. Parse raw LLM output into JSON (handle common failure modes)
# ---------------------------------------------------------------------------

def extract_json_array(raw: str) -> List[Dict[str, Any]]:
    """
    Extract a JSON array from raw LLM output, handling:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose
    - Trailing commas (common LLM error)
    - Truncated output (attempts to close incomplete arrays)
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Find the outermost [ ... ] using depth tracking (not naive rfind)
    # Rationale: rfind("]") fails when ] appears inside JSON string values.
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM output")

    depth = 0
    in_str = False
    esc = False
    end = None
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
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
        json_str = cleaned[start:end]
    else:
        # Rationale: if array is incomplete (truncated), attempt to close it gracefully.
        json_str = cleaned[start:]
        if in_str:
            json_str += '"'
        brace_depth = json_str.count('{') - json_str.count('}')
        for _ in range(brace_depth):
            json_str += '}'
        json_str += ']'

    # Fix trailing commas before ] or }
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json.loads(json_str)


# ---------------------------------------------------------------------------
# 2. Sentence-level deduplication within a single text field
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (handles common abbreviations)."""
    # Simple sentence splitter — works for clinical text
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def deduplicate_text(
    text: str,
    similarity_threshold: float = 0.75,
) -> str:
    """
    Remove duplicate or near-duplicate sentences from a text block.

    Parameters
    ----------
    text : str
        Input text (e.g., an alert message).
    similarity_threshold : float
        Sentences with similarity >= this value to any earlier sentence are removed.

    Returns
    -------
    str
        Deduplicated text.
    """
    sentences = _split_sentences(text)
    kept = []

    for sentence in sentences:
        is_duplicate = False
        for existing in kept:
            if _similarity(sentence, existing) >= similarity_threshold:
                is_duplicate = True
                logger.debug(f"Removed duplicate sentence: '{sentence[:80]}...'")
                break
        if not is_duplicate:
            kept.append(sentence)

    return " ".join(kept)


# ---------------------------------------------------------------------------
# 3. Paragraph-level deduplication (catches copy-pasted blocks)
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> List[str]:
    """Split on paragraph markers: double newline, pipe separators, or 'Evidence' headers."""
    parts = re.split(r"\n\n+|\s*\|\s*", text)
    return [p.strip() for p in parts if p.strip()]


def deduplicate_paragraphs(
    text: str,
    similarity_threshold: float = 0.70,
) -> str:
    """Remove near-duplicate paragraphs from text."""
    paragraphs = _split_paragraphs(text)
    kept = []

    for para in paragraphs:
        is_duplicate = False
        for existing in kept:
            if _similarity(para, existing) >= similarity_threshold:
                is_duplicate = True
                logger.debug(f"Removed duplicate paragraph: '{para[:80]}...'")
                break
        if not is_duplicate:
            kept.append(para)

    return " | ".join(kept) if len(kept) > 1 else " ".join(kept)


# ---------------------------------------------------------------------------
# 4. Alert-level deduplication (catches duplicate alerts entirely)
# ---------------------------------------------------------------------------

def deduplicate_alerts(
    alerts: List[Dict[str, Any]],
    similarity_threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    """
    Remove alerts that are near-duplicates of earlier alerts
    (based on title + message similarity).
    """
    kept = []

    for alert in alerts:
        alert_text = f"{alert.get('title', '')} {alert.get('message', '')}"
        is_duplicate = False

        for existing in kept:
            existing_text = f"{existing.get('title', '')} {existing.get('message', '')}"
            if _similarity(alert_text, existing_text) >= similarity_threshold:
                is_duplicate = True
                # Merge citations from duplicate into the kept alert
                existing_citations = {
                    c.get("chunk_id", "") for c in existing.get("citations", []) if isinstance(c, dict)
                }
                for citation in alert.get("citations", []):
                    if isinstance(citation, dict) and citation.get("chunk_id", "") not in existing_citations:
                        existing.setdefault("citations", []).append(citation)
                logger.info(
                    f"Merged duplicate alert '{alert.get('alert_id')}' "
                    f"into '{existing.get('alert_id')}'"
                )
                break

        if not is_duplicate:
            kept.append(alert)

    return kept


# ---------------------------------------------------------------------------
# 5. Enforce word limits
# ---------------------------------------------------------------------------

def enforce_word_limit(text: str, max_words: int) -> str:
    """Truncate text to max_words, ending at a sentence boundary if possible."""
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    # Try to end at the last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.5:  # Only if we keep at least half
        truncated = truncated[: last_period + 1]

    logger.warning(
        f"Truncated message from {len(words)} to {len(truncated.split())} words"
    )
    return truncated


# ---------------------------------------------------------------------------
# 6. Validate alert schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"alert_id", "title", "message", "issue_type", "severity", "recommended_action", "citations"}
VALID_ISSUE_TYPES = {
    "medication_safety", "monitoring_gap", "toxicity", "drug_interaction",
    "guideline_deviation", "plan_review", "plan_add", "information_gap",
}
VALID_SEVERITIES = {"critical", "high", "moderate", "low"}


def validate_alert(alert: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty if valid)."""
    errors = []

    missing = REQUIRED_KEYS - set(alert.keys())
    if missing:
        errors.append(f"Missing keys: {missing}")

    if alert.get("issue_type") not in VALID_ISSUE_TYPES:
        errors.append(f"Invalid issue_type: {alert.get('issue_type')}")

    if alert.get("severity") not in VALID_SEVERITIES:
        errors.append(f"Invalid severity: {alert.get('severity')}")

    citations = alert.get("citations", [])
    if alert.get("issue_type") == "information_gap":
        if citations:
            errors.append("information_gap alerts must have empty citations")
    else:
        if not citations:
            errors.append(f"Non-information_gap alert missing citations")

    return errors


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def clean_artemis_output(
    raw_response: str,
    message_max_words: int = 200,
    action_max_words: int = 50,
    sentence_similarity: float = 0.75,
    paragraph_similarity: float = 0.70,
    alert_similarity: float = 0.70,
) -> List[Dict[str, Any]]:
    """
    Full post-processing pipeline for ARTEMIS LLM output.

    Parameters
    ----------
    raw_response : str
        Raw string from the LLM API call.
    message_max_words : int
        Maximum words for the message field.
    action_max_words : int
        Maximum words for the recommended_action field.
    sentence_similarity : float
        Threshold for sentence-level dedup.
    paragraph_similarity : float
        Threshold for paragraph-level dedup.
    alert_similarity : float
        Threshold for alert-level dedup.

    Returns
    -------
    list[dict]
        Cleaned, validated alerts.
    """
    # Step 1: Parse JSON
    try:
        alerts = extract_json_array(raw_response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse LLM output: {e}")
        return [{
            "alert_id": "PARSE_ERROR",
            "title": "System: Failed to parse ARTEMIS output",
            "message": f"The LLM output could not be parsed as valid JSON. Error: {str(e)}. Raw output length: {len(raw_response)} characters.",
            "issue_type": "information_gap",
            "severity": "high",
            "recommended_action": "Retry the ARTEMIS analysis or review raw output manually.",
            "citations": [],
        }]

    if not isinstance(alerts, list):
        alerts = [alerts]

    # Step 2: Deduplicate within each alert's text fields
    for alert in alerts:
        if "message" in alert:
            alert["message"] = deduplicate_paragraphs(
                alert["message"], paragraph_similarity
            )
            alert["message"] = deduplicate_text(
                alert["message"], sentence_similarity
            )
            alert["message"] = enforce_word_limit(
                alert["message"], message_max_words
            )

        if "recommended_action" in alert:
            alert["recommended_action"] = deduplicate_text(
                alert["recommended_action"], sentence_similarity
            )
            alert["recommended_action"] = enforce_word_limit(
                alert["recommended_action"], action_max_words
            )

    # Step 3: Deduplicate across alerts
    alerts = deduplicate_alerts(alerts, alert_similarity)

    # Step 4: Validate
    valid_alerts = []
    for alert in alerts:
        errors = validate_alert(alert)
        if errors:
            logger.warning(
                f"Alert '{alert.get('alert_id', 'UNKNOWN')}' has validation issues: {errors}"
            )
        valid_alerts.append(alert)  # Keep but log — don't silently drop

    logger.info(
        f"Post-processing complete: {len(valid_alerts)} alerts "
        f"(from {len(alerts)} raw)"
    )
    return valid_alerts
